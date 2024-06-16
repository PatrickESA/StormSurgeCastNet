import os
import sys

import torch
import torch.nn as nn

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(dirname), 'util'))
import losses, model_utils



class BaseModel(nn.Module):
    def __init__(
        self,
        config
    ):
        super(BaseModel, self).__init__()
        self.config     = config                    # store config
        self.frozen     = False                     # no parameters are frozen
        self.len_epoch  = 0                         # steps of one epoch

        # define embedder of lead time conditioning vector
        if self.config.film:
            self.lead_time_embedding = nn.Embedding(self.config.max_lead_times, self.config.film_latent).to(self.config.device)
        else: self.lead_time_embedding = None

        # fetch generator
        self.netG = model_utils.get_generator(self.config)

        # 1 criterion
        self.criterion = losses.get_loss(self.config)

        # 2 optimizer: for G
        paramsG = [{'params': self.netG.parameters()}]

        self.optimizer_G = torch.optim.Adam(paramsG, lr=config.lr)

        # 2 scheduler: for G, note: stepping takes place at the end of epoch
        self.scheduler_G = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_G, gamma=self.config.gamma)

        self.real_A = None # the input in the context A -> B
        self.fake_B = None # predictions
        self.real_B = None # targets
        self.dates  = None # don't need this for Metnet dont pass the dates
        self.masks  = None # extra channel mask or could be the land sea mask
        self.lead   = None # lead data

    def forward(self):
        # forward through generator, note: for val/test splits, x
        # 'with torch.no_grad():' is declared in train script
        if self.config.film and self.lead is not None: 
            # get [B x 1 x latentDim] embedding matrix
            cond = self.lead_time_embedding(self.lead.int())
        else: cond = None
        if self.config.model == "metnet3":
            self.fake_B = self.netG(X=self.real_A, lead_times=cond)
        else:
            self.fake_B = self.netG(self.real_A, batch_positions=self.dates, lead=cond)
    
    def backward_G(self):
        # calculate generator loss
        self.get_loss_G()
        self.loss_G.backward()


    def get_loss_G(self):
        self.loss_G = losses.calc_loss(self.criterion, self.config, self.fake_B, self.real_B)

    def set_input(self, input):
        self.real_A = input['A'].to(self.config.device)
        self.real_B = input['B'].to(self.config.device)
        self.dates  = None if input['dates'] is None else input['dates'].to(self.config.device)
        self.masks  = input['masks'].to(self.config.device)
        if self.config.film and 'lead' in input: self.lead = input['lead'].to(self.config.device)
    
    def reset_input(self):
        self.real_A = None
        self.real_B = None
        self.dates  = None 
        self.masks  = None
        self.lead   = None
        del self.real_A
        del self.real_B 
        del self.dates
        del self.masks
        #del self.lead

    def optimize_parameters(self):
        self.forward()
        del self.real_A

        # update G
        self.optimizer_G.zero_grad() 
        self.backward_G()
        self.optimizer_G.step()
        # resetting inputs after optimization saves memory
        self.reset_input()

        if self.netG.training: 
            self.fake_B = self.fake_B.cpu()