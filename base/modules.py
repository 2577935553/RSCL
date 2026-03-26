import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)


class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


class Contrastive_Loss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super(Contrastive_Loss, self).__init__()

        self.tau = temperature

    def forward(self, input, positive, negative):
        B, C, *spatial_size = input.shape
        spatial_dim = len(spatial_size)
        norm_input = F.normalize(input.permute([0, *list(range(2, 2 + spatial_dim)), 1]).reshape(-1, C), dim=-1)
        norm_positive = F.normalize(positive.permute([0, *list(range(2, 2 + spatial_dim)), 1]).reshape(-1, C), dim=-1)
        norm_negative = F.normalize(negative.permute([0, *list(range(2, 2 + spatial_dim)), 1]).reshape(-1, C), dim=-1)

        numerator = F.cosine_similarity(norm_input, norm_positive, dim=-1)
        negative_similarity = F.cosine_similarity(norm_input, norm_negative, dim=-1)
        nominator = torch.exp(numerator / self.tau)
        denominator = torch.exp(negative_similarity / self.tau).sum(0) + nominator
        loss = -torch.log(nominator / (denominator + 1e-7)).mean()
        return loss
def posMask(pred1, pred2, class_num):
    B,C,H,W = pred1.shape
    pred1 = pred1.reshape(B, H*W) # B,HW,N
    pred2 = pred2.reshape(B, H*W)

    pred1_one_hot = F.one_hot(pred1.long(), num_classes=class_num).float().cuda()
    pred2_one_hot = F.one_hot(pred2.long(), num_classes=class_num).float().cuda()
    mask_p = torch.bmm(pred1_one_hot, pred2_one_hot.transpose(1,2))
    return mask_p
def negMask(pred1, pred2, class_num):
    B, C, H, W = pred1.shape
    pred1 = pred1.reshape(B, H * W)
    pred2 = pred2.reshape(B, H * W)

    pred1_one_hot = F.one_hot(pred1.long(), num_classes=class_num).float().cuda()
    pred2_one_hot = F.one_hot(pred2.long(), num_classes=class_num).float().cuda()
    mask_p = torch.bmm(pred1_one_hot, pred2_one_hot.transpose(1, 2))
    return 1 - mask_p
def regression_loss(pred, pos_pred_1, neg_pred_1, neg_pred_2, label_pred, label_pos_pred1, label_neg_pred1, label_neg_pred2, class_num):
    N,C,H,W = pred.shape
    pred = pred.view(N, C, -1)
    pos_pred_1 = pos_pred_1.view(N, C,-1)
    neg_pred_1 = neg_pred_1.view(N, C, -1)
    neg_pred_2 = neg_pred_2.view(N, C, -1)

    logits1 = torch.bmm(pred.transpose(1, 2), pos_pred_1)
    logits2 = torch.bmm(pred.transpose(1, 2), neg_pred_1)
    logits3 = torch.bmm(pred.transpose(1, 2), neg_pred_2)

    mask_p = posMask(label_pred, label_pos_pred1, class_num).cuda().detach()
    mask_p2 = posMask(label_pred, label_neg_pred1, class_num).cuda().detach()
    mask_p3 = posMask(label_pred, label_neg_pred2, class_num).cuda().detach()

    mask_n = negMask(label_pred, label_pos_pred1, class_num).cuda().detach()
    mask_n2 = negMask(label_pred, label_neg_pred1, class_num).cuda().detach()
    mask_n3 = negMask(label_pred, label_neg_pred2, class_num).cuda().detach()

    masked_p = mask_p * logits1
    masked_p2 = mask_p2 * logits2
    masked_p3 = mask_p3 * logits3

    masked_n = mask_n * logits1
    masked_n2 = mask_n2 * logits2
    masked_n3 = mask_n3 * logits3

    P = torch.sum(masked_p + masked_p2+masked_p3, dim=-1) / (torch.sum(mask_p+mask_p2+mask_p3, dim=-1) + 1e-6)
    N = torch.sum(masked_n, dim=-1) / (torch.sum(mask_n, dim=-1)+ 1e-6) + torch.sum(masked_n2, dim=-1) / (torch.sum(mask_n2, dim=-1)+ 1e-6) \
        + torch.sum(masked_n3, dim=-1) / (torch.sum(mask_n3, dim=-1)+ 1e-6)
    P_exp = torch.exp(P)
    N_exp = torch.exp(N)
    loss = -torch.mean( torch.log(P_exp / (P_exp + N_exp)+1e-6))
    return loss

class Contras_Loss_v2(nn.Module):
    def __init__(self, num_classes):
        super(Contras_Loss_v2, self).__init__()
        self.num_classes = num_classes
    def forward(self, pred, pred_pos_1, pred_neg_1, pred_neg_2, mask, mask_pos, mask_neg_1, mask_neg_2):
        N,C,H,W = pred.shape
        pred = F.normalize(pred, dim=1)
        pred_pos_1 = F.normalize(pred_pos_1, dim=1)
        pred_neg_1 = F.normalize(pred_neg_1, dim=1)
        pred_neg_2 = F.normalize(pred_neg_2, dim=1)
        mask = F.interpolate(mask.float(), size=[H,W], mode='nearest')
        mask_pos = F.interpolate(mask_pos.float(), size=[H,W], mode='nearest')
        mask_neg_1 = F.interpolate(mask_neg_1.float(), size=[H,W], mode='nearest')
        mask_neg_2 = F.interpolate(mask_neg_2.float(), size=[H,W], mode='nearest')

        loss = regression_loss(pred, pred_pos_1, pred_neg_1, pred_neg_2, mask, mask_pos, mask_neg_1, mask_neg_2, self.num_classes)
        return loss
def uncertainty_loss(inputs, targets):
    """
    Uncertainty rectified pseudo supervised loss
    """
    # detach from the computational graph
    pseudo_label = torch.argmax(F.softmax(targets, dim=1).detach(), dim=1)
    vanilla_loss = F.cross_entropy(inputs, pseudo_label.long(), reduction='none')
    # uncertainty rectification
    cos_similarity = F.cosine_similarity(inputs, targets.detach(), dim=1)

    uncertainty_loss = (torch.exp(cos_similarity - 1) * vanilla_loss).mean() + (1 - cos_similarity).mean()
    return uncertainty_loss


def uncertainty_loss_mse(inputs, targets):
    """
    with mse
    :param inputs:
    :param targets:
    :return:
    """
    pseudo_label = torch.argmax(F.softmax(targets, dim=1).detach(), dim=1)
    vanilla_loss = F.cross_entropy(inputs, pseudo_label.long(), reduction='none')
    print(vanilla_loss.shape)
    # uncertainty rectification
    # mse_similarity = F.mse_loss(inputs, targets.detach())
    mae_similarity = torch.mean(torch.abs(inputs - targets.detach()), dim=1)
    uncertainty_loss = (torch.exp(mae_similarity - 1) * vanilla_loss).mean() + (1 - mae_similarity).mean()
    return uncertainty_loss


def uncertainty_loss_kl(inputs, targets):
    """
    with mse
    :param inputs:
    :param targets:
    :return:
    """
    pseudo_label = torch.argmax(F.softmax(targets, dim=1).detach(), dim=1)
    vanilla_loss = F.cross_entropy(inputs, pseudo_label.long(), reduction='none')
    # uncertainty rectification
    kl_div = torch.sum(F.kl_div(F.log_softmax(inputs, dim=1), F.softmax(targets, dim=1).detach(), reduction='none'),
                       dim=1)
    uncertainty_loss = (torch.exp(-kl_div) * vanilla_loss).mean() + kl_div.mean()
    return uncertainty_loss


if __name__ == '__main__':
    inputs = torch.rand(4, 4, 224, 224)
    pseudo = torch.rand(4, 4, 224, 224)
    # vanilla_loss = F.cross_entropy(inputs, inputs, reduction='none')
    # kl = F.kl_div(F.log_softmax(inputs, dim=1), F.softmax(inputs, dim=1).detach(), reduction='none')
    print(uncertainty_loss_mse(inputs, pseudo))
