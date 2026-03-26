import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import csv

def crop_image(image, cx, cy, size, constant_values=0):
    """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
    X, Y = image.shape[:2]
    rX = size[0] // 2
    rY = size[1] // 2
    x1, x2 = cx - rX, cx + (size[0] - rX)
    y1, y2 = cy - rY, cy + (size[1] - rY)
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    # Crop the image
    crop = image[x1_: x2_, y1_: y2_]
    # Pad the image if the specified size is larger than the input image size
    if crop.ndim == 3:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)),
                      'constant', constant_values=constant_values)
    elif crop.ndim == 4:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)),
                      'constant', constant_values=constant_values)
    else:
        print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
        exit(0)
    return crop


# def calculate_metric_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     if pred.sum() > 0:
#         dice = metric.binary.dc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#         return dice, hd95
#     else:
#         return 0, 0
    
def calculate_metric_percase(pred, gt):

    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0
    
    if pred.sum() == 0:
        return 0.0, 100.0  # 返回最差的分数
    
    if gt.sum() == 0:
        return 0.0, 100.0  # 返回最差的分数
    
    try:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    except Exception as e:
        print(f"Metric calculation failed: {e}")
        return 0.0, 100.0


def test_single_volume(image, label, net, classes, patch_size=None,api_md=None,api_of=False,api_of_half=False):
    if patch_size is None:
        patch_size = [256, 256]
    image, label = image.squeeze().cpu().detach(
    ).numpy(), label.squeeze().cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        api_md.eval()
        with torch.no_grad():
            out=net(input)[0]
            if api_of:
                assert api_md is not None
                out=api_md(out)
            if api_of_half:
                out=(out+api_md(out))/2
            
            out = torch.argmax(torch.softmax(
                out, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = out
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume_DNCC(image, label, net, classes, patch_size=None):
    if patch_size is None:
        patch_size = [256, 256]
    image, label = image.squeeze().cpu().detach(
    ).numpy(), label.squeeze().cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out=net(input)[0]
            out = torch.argmax(torch.softmax(
                out, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = out
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

# def test_single_volume_UCC(image, label, net, classes, patch_size=None):
#     if patch_size is None:
#         patch_size = [224, 224]
#     image, label = image.squeeze().cpu().detach(
#     ).numpy(), label.squeeze().cpu().detach().numpy()
#     prediction = np.zeros_like(label)
#     for ind in range(image.shape[0]):
#         slice = image[ind, :, :]
#         x, y = slice.shape[0], slice.shape[1]
#         input = torch.from_numpy(slice).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(
#                 net(input)['predictions'], dim=1), dim=1).squeeze(0)
#             out = out.cpu().detach().numpy()
#             pred = out
#             prediction[ind] = pred
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(
#             prediction == i, label == i))
#     return metric_list

def test_single_volume_UCC(image, label, net, classes, patch_size=None):
    """
    测试单个volume的性能，处理UCC网络的输出格式
    """
    if patch_size is None:
        patch_size = [224, 224]
    
    image = image.squeeze().cpu().detach().numpy()
    label = label.squeeze().cpu().detach().numpy()
    
    prediction = np.zeros_like(label)
    
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        
        net.eval()
        with torch.no_grad():
            output = net(input)
            
            if isinstance(output, dict):
                pred_logits = output['predictions']
            else:
                pred_logits = output
            
            out = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction[ind] = out
    
    metric_list = []
    for i in range(1, classes):
        pred_i = (prediction == i).astype(np.float32)
        label_i = (label == i).astype(np.float32)
        
        dice, hd95 = calculate_metric_percase(pred_i, label_i)
        metric_list.append([dice, hd95])
    
    return metric_list

def test_single_volume_ds(image, label, net, classes, patch_size=None):
    if patch_size is None:
        patch_size = [256, 256]
    image, label = image.squeeze().cpu().detach(
    ).numpy(), label.squeeze().cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def get_image_list(csv_file):
    image_list, label_list = [], []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_list.append(row['image_filenames'].replace('\t', '').strip())
            label_list.append(row['label_filenames'].replace('\t', '').strip())

    data_list = {}
    data_list['image_filenames'] = image_list
    data_list['label_filenames'] = label_list

    return data_list