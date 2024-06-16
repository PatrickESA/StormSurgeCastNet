import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


class Metric(object):
    """Base class for all metrics.
    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """
    
    def reset(self): pass
    def add(self):   pass
    def value(self): pass


def img_metrics(target, pred):
    rmse = torch.sqrt(torch.mean(torch.square(target - pred)))
    mae = torch.mean(torch.abs(target - pred))

    metric_dict = {'RMSE': rmse.cpu().numpy().item(),
                   'MAE': mae.cpu().numpy().item()}
    
    return metric_dict

class avg_img_metrics(Metric):
    def __init__(self):
        super().__init__()
        self.n_samples = 0
        self.metrics   = ['RMSE', 'MAE']
        self.metrics  += ['error', 'mean se', 'mean ae']

        self.running_img_metrics = {}
        self.running_nonan_count = {}
        self.reset()

    def reset(self):
        for metric in self.metrics: 
            self.running_nonan_count[metric] = 0
            self.running_img_metrics[metric] = np.nan

    def add(self, metrics_dict):
        for key, val in metrics_dict.items():
            # skip variables not registered
            if key not in self.metrics: continue
            # filter variables not translated to numpy yet
            if torch.is_tensor(val): continue
            if isinstance(val, tuple): val=val[0]

            # only keep a running mean of non-nan values
            if np.isnan(val): continue

            if not self.running_nonan_count[key]: 
                self.running_nonan_count[key] = 1
                self.running_img_metrics[key] = val
            else: 
                self.running_nonan_count[key]+= 1
                self.running_img_metrics[key] = (self.running_nonan_count[key]-1)/self.running_nonan_count[key] * self.running_img_metrics[key] \
                                                + 1/self.running_nonan_count[key] * val

    def value(self):
        return self.running_img_metrics