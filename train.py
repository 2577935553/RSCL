import torch
import tqdm
import argparse
from torch.utils.data import DataLoader
from SegModel import ProjectUNet1
import random
import os
import numpy as np
from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from utilities import losses
import torch.nn.functional as F
from utilities.val_2D import test_single_volume_DNCC, get_image_list, crop_image
import nibabel as nib
from utilities.MyDataSet import SemiSegDataset_2, SemiSegDataset_1
from utilities.Load_Data_v2 import augment_data_batch
from rscl import RSCL
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# torch.autograd.set_detect_anomaly(True)  # enable only for debugging


def create_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=0):
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    return scheduler


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/public/home/zhsy/data/zhsy_data/training')
    parser.add_argument("--train_data_csv", type=str, default='./data_dir/train_ACDC.csv')
    parser.add_argument("--valid_data_csv", type=str, default='./data_dir/valid_ACDC.csv')
    parser.add_argument("--train_output_dir", type=str, default='/public/home/zhsy/data/zhsy_data/RSCL/ACDC')
    parser.add_argument("--test_output_dir", type=str, default='/public/home/zhsy/data/zhsy_data/RSCL/ACDC')
    parser.add_argument("--test_data_dir", type=str, default='/public/home/zhsy/data/zhsy_data/testing')
    parser.add_argument("--test_data_list", type=str, default='./data_dir/test_subj_ACDC.csv')
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--label_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--image_size", nargs='+', type=int, default=[224, 224])
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    # RSCL hyperparameters
    parser.add_argument("--lambda_dgpc", type=float, default=0.2)
    parser.add_argument("--lambda_ucps", type=float, default=1.0)
    parser.add_argument("--tau_high_final", type=float, default=0.85)
    parser.add_argument("--tau_low_final", type=float, default=0.30)
    parser.add_argument("--tau_soft", type=float, default=0.5)
    # Augmentation control: 'acdc' or 'camus'
    parser.add_argument("--aug_mode", type=str, default='acdc', choices=['acdc', 'camus','la'])
    # Ablation control
    parser.add_argument("--contrastive_mode", type=str, default='dual',
                        choices=['dual', 'all_hard', 'all_hard_filtered', 'hard_only', 'soft_only'],
                        help="Dual-granularity ablation: which zones get which loss")
    parser.add_argument("--baseline_cps", action='store_true', default=False,
                        help="Use standard unweighted CPS instead of UCPS (for vanilla baseline)")
    parser.add_argument("--ensemble", action='store_true', default=False)
    parser.add_argument("--title", type=str, default='')
    return parser.parse_args()


def training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_torch(args.seed)
    g_new_worker = torch.Generator()
    g_new_worker.manual_seed(args.seed)

    train_output_dir = args.train_output_dir + args.title
    model_dir = train_output_dir + '/model'
    training_graph = train_output_dir + '/graph'
    mkdir(train_output_dir)
    mkdir(model_dir)
    mkdir(training_graph)

    batch_size = args.batch_size
    image_size = args.image_size
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    warmup_iter = 1000  # RSCL warmup: only sup+ucps for first 1000 iters

    # Augmentation params per dataset
    if args.aug_mode == 'acdc':
        aug_rot_l, aug_scale_l = 60, 0.5
        aug_rot_u, aug_scale_u = 30, 0.2
        shift_label, shift_unlabel = 60, 30
    elif args.aug_mode == 'camus':  # camus
        aug_rot_l, aug_scale_l = 30, 0.2
        aug_rot_u, aug_scale_u = 30, 0.2
        shift_label, shift_unlabel = 60, 30
    elif args.aug_mode == 'la':  # camus
        aug_rot_l, aug_scale_l = 30, 0.2
        aug_rot_u, aug_scale_u = 30, 0.2
        shift_label, shift_unlabel = 15, 15
    else:
        assert False

    # Dataset
    train_set = SemiSegDataset_2(
        args.data_dir, args.train_data_csv, args.valid_data_csv, image_size,
        label_ratio=args.label_ratio, mode='train', random_seed=args.seed,
        shift_label=shift_label, shift_unlabel=shift_unlabel)
    valset = SemiSegDataset_1(
        args.data_dir, args.train_data_csv, args.valid_data_csv, image_size,
        label_ratio=args.label_ratio, mode='valid', random_seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0,
                              drop_last=True, shuffle=True, generator=g_new_worker)
    val_loader = DataLoader(valset, batch_size=20, shuffle=False)

    # Models (identical to reference)
    model1 = ProjectUNet1('resnet50', None, classes=num_classes, deep_stem=32).cuda()
    model2 = ProjectUNet1('resnet50', None, classes=num_classes, deep_stem=32).cuda()

    # RSCL module
    rscl = RSCL(num_classes=num_classes, feat_dim=128,
                contrastive_mode=args.contrastive_mode,
                tau_high_final=args.tau_high_final,
                tau_low_final=args.tau_low_final,
                tau_soft=args.tau_soft).cuda()

    writer = SummaryWriter(training_graph)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    # Optimizers (same lr scheme as reference: base_lr * 0.04)
    lr = args.learning_rate * 0.04
    optimizer1 = optim.AdamW(model1.parameters(), lr=lr, weight_decay=0.0001)
    optimizer2 = optim.AdamW(model2.parameters(), lr=lr, weight_decay=0.0001)

    warmup_epochs = int(max_iterations * 0.1)
    scheduler1 = create_warmup_cosine_scheduler(optimizer1, warmup_epochs, max_iterations)
    scheduler2 = create_warmup_cosine_scheduler(optimizer2, warmup_epochs, max_iterations)

    iter_num = 0
    best_performance1 = 0.0

    iterator = tqdm.tqdm(range(args.epochs), ncols=70)
    for epoch in iterator:
        print("Epoch {}/{}, lr {:.6f}".format(epoch + 1, args.epochs, optimizer1.param_groups[0]['lr']))

        for step, (batch_labeled_data, batch_labeled_lab, batch_unlabeled_data) in enumerate(train_loader):
            model1.train()
            model2.train()

            # Data augmentation (same as reference)
            aug_labeled_img, aug_labeled_lab = batch_labeled_data.numpy().squeeze(), batch_labeled_lab.numpy()
            aug_labeled_img, aug_labeled_lab = augment_data_batch(
                aug_labeled_img, aug_labeled_lab, shift=0, rotate=aug_rot_l, scale=aug_scale_l, flip=False)
            aug_labeled_img = torch.Tensor(aug_labeled_img).unsqueeze(1)
            aug_labeled_lab = torch.Tensor(aug_labeled_lab).long()

            aug_unlabeled_img = batch_unlabeled_data.flatten(0, 1).numpy().squeeze()
            aug_unlabeled_img, _ = augment_data_batch(
                aug_unlabeled_img, aug_unlabeled_img, shift=0, rotate=aug_rot_u, scale=aug_scale_u, flip=False)
            aug_unlabeled_img = torch.Tensor(aug_unlabeled_img).unsqueeze(1)

            image_labeled = aug_labeled_img.to(device)
            label_labeled = aug_labeled_lab.to(device)
            image_unlabeled = aug_unlabeled_img.to(device)

            # Forward: y=logits, dx=decoder_feat(128ch), x=encoder_feat
            y_l1, dx_l1, _ = model1(image_labeled)
            y_u1, dx_u1, _ = model1(image_unlabeled)
            y_l2, dx_l2, _ = model2(image_labeled)
            y_u2, dx_u2, _ = model2(image_unlabeled)

            # Supervised loss (both networks, same as reference)
            out_l_soft1 = F.softmax(y_l1, dim=1)
            out_l_soft2 = F.softmax(y_l2, dim=1)
            loss_sup1 = 0.5 * ce_loss(y_l1, label_labeled) + dice_loss(out_l_soft1, label_labeled)
            loss_sup2 = 0.5 * ce_loss(y_l2, label_labeled) + dice_loss(out_l_soft2, label_labeled)
            loss_sup = loss_sup1 + loss_sup2

            # RSCL losses
            loss_dgpc, loss_ucps = rscl(
                dx_l1, dx_l2, dx_u1, dx_u2,
                y_l1, y_l2, y_u1, y_u2,
                label_labeled, iter_num, max_iterations)

            # Standard CPS baseline: unweighted cross pseudo supervision
            if args.baseline_cps:
                pseudo2 = y_u2.detach().argmax(dim=1)
                pseudo1 = y_u1.detach().argmax(dim=1)
                loss_ucps = (F.cross_entropy(y_u1, pseudo2) + F.cross_entropy(y_u2, pseudo1)) / 2.0

            # Ramp-up for UCPS (sigmoid over first 2000 iters)
            rampup = min(1.0, iter_num / 2000.0)

            # Total loss
            if iter_num >= warmup_iter:
                total_loss = loss_sup + args.lambda_dgpc * loss_dgpc + args.lambda_ucps * rampup * loss_ucps
            else:
                total_loss = loss_sup + args.lambda_ucps * rampup * loss_ucps

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            total_loss.backward()
            optimizer1.step()
            optimizer2.step()

            iter_num += 1
            scheduler1.step()
            scheduler2.step()

            # Logging
            writer.add_scalar('lr', scheduler1.get_last_lr()[0], iter_num)
            writer.add_scalar('loss/total', total_loss.item(), iter_num)
            writer.add_scalar('loss/sup', loss_sup.item(), iter_num)
            writer.add_scalar('loss/dgpc', loss_dgpc.item(), iter_num)
            writer.add_scalar('loss/ucps', loss_ucps.item(), iter_num)

            iterator.desc = "Epoch[{}/{}] sup:{:.3f} dgpc:{:.3f} ucps:{:.3f}".format(
                epoch, args.epochs, loss_sup.item(), loss_dgpc.item(), loss_ucps.item())

            if iter_num % 50 == 0:
                image = image_labeled[0, 0:1, :, :]
                writer.add_image('train/Image', image.cpu(), iter_num)
                outputs = torch.argmax(F.softmax(y_l1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_pred', outputs[0].float().cpu() * 50, iter_num)
                gt = label_labeled[0].unsqueeze(0)
                writer.add_image('train/GT', gt.float().cpu(), iter_num)

            # Validation
            if iter_num > 0 and iter_num % 50 == 0:
                model1.eval()
                metric_list = 0.0
                for sample_batch in val_loader:
                    metric_i = test_single_volume_DNCC(
                        sample_batch[0], sample_batch[1], model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list /= len(val_loader)

                for c in range(num_classes - 1):
                    writer.add_scalar('val/class_{}_dice'.format(c + 1), metric_list[c, 0], iter_num)
                    writer.add_scalar('val/class_{}_hd95'.format(c + 1), metric_list[c, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('val/mean_dice', performance1, iter_num)
                print('iter {} | Avg DICE: {:.4f}'.format(iter_num, performance1))

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_best = os.path.join(model_dir, 'best_model1.pth')
                    torch.save(model1.state_dict(), save_best)
                    if args.ensemble:
                        torch.save(model2.state_dict(), save_best.replace('model1', 'model2'))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break

    writer.close()


def testing(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_torch(args.seed)
    image_size = args.image_size
    num_classes = args.num_classes

    train_output_dir = args.train_output_dir + args.title
    model_dir = train_output_dir + '/model'
    test_output_dir = args.test_output_dir + args.title
    pred_dir = test_output_dir + '/predictions'
    mkdir(pred_dir)

    model = ProjectUNet1('resnet50', None, classes=num_classes, deep_stem=32).cuda()
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model1.pth'), map_location=device))
    model.eval()

    model2 = None
    if args.ensemble:
        model2 = ProjectUNet1('resnet50', None, classes=num_classes, deep_stem=32).cuda()
        model2.load_state_dict(torch.load(os.path.join(model_dir, 'best_model2.pth'), map_location=device))
        model2.eval()

    data_list = get_image_list(args.test_data_list)
    print("Testing on {} images".format(len(data_list['image_filenames'])))

    with torch.no_grad():
        for index in range(len(data_list['image_filenames'])):
            gt_name = data_list['label_filenames'][index]
            nib_gt = nib.load(os.path.join(args.test_data_dir, gt_name))
            gt = nib_gt.get_fdata()

            img_name = data_list['image_filenames'][index]
            nib_img = nib.load(os.path.join(args.test_data_dir, img_name))
            img = nib_img.get_fdata().astype('float32')

            clip_min = np.percentile(img, 1)
            clip_max = np.percentile(img, 99)
            img = np.clip(img, clip_min, clip_max)
            img = (img - img.min()) / float(img.max() - img.min())
            x, y, z = img.shape
            x_centre, y_centre = int(x / 2), int(y / 2)
            img = crop_image(img, x_centre, y_centre, image_size, constant_values=0)

            pred_res = torch.zeros(img.shape, dtype=torch.int8)
            for i in range(img.shape[2]):
                tmp_image = torch.from_numpy(img[:, :, i]).unsqueeze(0).unsqueeze(0).to(device)
                outputs = model(tmp_image)[0]
                if args.ensemble and model2 is not None:
                    outputs = (outputs + model2(tmp_image)[0]) / 2
                pred_res[:, :, i] = torch.argmax(F.softmax(outputs, dim=1)[0], dim=0)

            pred_res = crop_image(pred_res.numpy(), image_size[0] // 2, image_size[1] // 2, (x, y), constant_values=0)
            pred_res = pred_res.astype('int16')
            nii_pred = nib.Nifti1Image(pred_res, None, header=nib_gt.header)

            loc_end = img_name.find('.')
            loc_start = img_name.rfind('/')
            savedirname = os.path.join(pred_dir, img_name[:loc_start])
            mkdir(savedirname)
            pred_name = savedirname + img_name[loc_start:loc_end] + '_Pred.nii.gz'
            nib.save(nii_pred, pred_name)

    print('Finished Testing!')


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.device)
    if args.mode == 'train':
        training(args)
        testing(args)
    elif args.mode == 'test':
        testing(args)
    else:
        raise NotImplementedError
