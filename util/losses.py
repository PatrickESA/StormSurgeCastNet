import numpy as np
import torch

def get_loss(config):
    if config.loss=="weighted-l1":
        # compute sparse loss via nanmean reduction, wrap into additional check (because the nanmean of all NaNs is still NaN)
        criterion1 = lambda pred, targ: torch.nanmean(torch.nn.functional.l1_loss(targ, pred, reduction='none'))
        wrapper    = lambda arg: 0 if torch.isnan(arg) else arg
        # compute coarse loss, then weight both components via a scalar coefficient
        if config.out_conv[-1] == 1: criterion = lambda pred, targ: wrapper(criterion1(pred[:,:,[0],...], targ[:,:,[0],...]))
        elif config.out_conv[-1] == 2:
            criterion   = lambda pred, targ: wrapper(criterion1(pred[:,:,[0],...], targ[:,:,[0],...])) + config.weighting*wrapper(criterion1(pred[:,:,[1],...], targ[:,:,[1],...]))
        else: raise NotImplementedError
    elif config.loss=="l1":
        criterion= lambda pred, targ: torch.nanmean(torch.nn.functional.l1_loss(targ, pred, reduction='none'))
    elif config.loss=="l2":
        criterion = lambda pred, targ: torch.nanmean(torch.nn.functional.mse_loss(targ, pred, reduction='none'))
    else: raise NotImplementedError

    # wrap losses
    loss_wrap = lambda *args: args
    loss = loss_wrap(criterion) 
    return loss if not isinstance(loss, tuple) else loss[0]


def calc_loss(criterion, config, out, y):
    return criterion(out, y)

# see https://github.com/neuralhydrology/neuralhydrology/blob/5436973d7645cf090d0d75b0ffd1d6b7f902cf68/neuralhydrology/evaluation/metrics.py#L52
def nnse(obs, sim):
    denominator = np.nansum(((obs - np.nanmean(obs))**2))
    numerator = np.nansum(((sim - obs)**2))

    value = 1 - numerator / denominator
    normalized_value = 1 / (2 - value)

    return float(normalized_value)