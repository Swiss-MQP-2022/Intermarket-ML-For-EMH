from torch import nn


class ComboLoss(nn.Module):
    def __init__(self,
                 c_bias=1,
                 regression_fn=nn.functional.mse_loss,
                 classification_fn=nn.functional.binary_cross_entropy):
        """
        :param c_bias: weight factor used to bias classification loss
        :param regression_fn: function for calculating regression loss
        :param classification_fn: function for calculating classification loss
        """

        super(ComboLoss, self).__init__()
        self.regression_fn = regression_fn
        self.classification_fn = classification_fn
        self.c_bias = c_bias

    def forward(self, reg_pred, reg_target, cla_pred, cla_target):
        reg_loss = self.regression_fn(reg_pred, reg_target)
        classification_loss = self.c_bias * self.classification_fn(cla_pred, cla_target)
        return reg_loss + classification_loss


class LimLundgrenLoss(nn.Module):
    def __init__(self, epsilon=0.01):
        super(LimLundgrenLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, truth):
        numer = (truth - pred).square().sum()
        denom = pred.sign().eq(truth.sign()).sum() + self.epsilon
        return numer / denom
