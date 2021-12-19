from torch import nn
import torch


class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.L1Loss()

    def forward(self, true, pred):
        loss = 0
        for t, p in zip(true, pred):
            for tl, pl in zip(t, p):
                loss += self.loss_func(tl, pl)
        return loss * 2


class GenLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disc_out):
        loss = 0
        for out in disc_out:
            l = torch.mean((1 - out) ** 2)
            loss += l

        return loss


class DiscLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disc_true_out, disc_pred_out):
        loss = 0
        for t, p in zip(disc_true_out, disc_pred_out):
            true_loss = torch.mean((1 - t) ** 2)
            pred_loss = torch.mean(p ** 2)
            loss += (true_loss + pred_loss)

        return loss
