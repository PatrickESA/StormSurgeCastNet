import os
import torch

from models import base_model
from models.utae.utae import UTAE
from models.metnet3.metnet3 import MetNet3
from models.lstm.models import LSTM, ConvLSTM


def get_base_model(config):
    model = base_model.BaseModel(config)
    return model

# for running image reconstruction
def get_generator(config):
    if "utae" in config.model:
        model = UTAE(
            input_dim=config.in_dim,
            encoder_widths=config.encoder_widths,
            decoder_widths=config.decoder_widths,
            out_conv=config.out_conv,
            out_nonlin_mean=config.mean_nonLinearity,
            str_conv_k=4,
            str_conv_s=2,
            str_conv_p=1,
            agg_mode=config.agg_mode,
            encoder_norm=config.encoder_norm,
            norm_skip='batch',
            norm_up='batch',
            decoder_norm=config.decoder_norm,
            n_head=config.n_head,
            d_model=config.d_model,
            d_k=config.d_k,
            encoder=False,
            return_maps=False,
            pad_value=config.pad_value,
            padding_mode=config.padding_mode,
            positional_encoding=config.positional_encoding,
            cond_dim=config.film_latent if config.film else None
        )
    elif "metnet3" == config.model:
        model = MetNet3(
            dim = 64,
            num_lead_times = 12,
            lead_time_embed_dim = 32,
            input_spatial_size = 256,
            attn_depth = 12,
            attn_dim_head = 64,
            attn_heads = config.n_head,
            attn_dropout = 0.1,
            vit_window_size = 8,
            vit_mbconv_expansion_rate = 4,
            vit_mbconv_shrinkage_rate = 0.25,
            input_2496_channels = 2 + 14 + 1 + 2 + 20,
            # input_4996_channels = 16 + 1,
            surface_and_hrrr_target_spatial_size = 128,
            precipitation_target_bins = dict(
                mrms_rate = 512,
                mrms_accumulation = 512
            ),
            surface_target_bins = dict(
                omo_temperature = 256,
                omo_dew_point = 256,
                omo_wind_speed = 256,
                omo_wind_component_x = 256,
                omo_wind_component_y = 256,
                omo_wind_direction = 180
            ),
            hrrr_norm_strategy = 'none',
            hrrr_channels = 256,
            hrrr_norm_statistics = None,
            hrrr_loss_weight = 10,
            # crop_size_post_16km = 48,
            resnet_block_depth = 2
        )
    elif 'lstm' == config.model:
        model = LSTM(in_channels=config.in_dim)
        assert config.hyperlocal, 'LSTM must be hyperlocal'
    elif 'conv_lstm' == config.model:
        model = ConvLSTM(in_channels=config.in_dim)
        assert config.hyperlocal, 'ConvLSTM must be hyperlocal'
    else: raise NotImplementedError
    return model


def get_model(config):
    return get_base_model(config)


def save_model(config, epoch, model, name):
    state_dict = {"epoch":          epoch,
                  "state_dict":     model.state_dict(),
                  "state_dict_G":   model.netG.state_dict(),
                  "optimizer_G":    model.optimizer_G.state_dict(),
                  "scheduler_G":    model.scheduler_G.state_dict()}
    torch.save(state_dict,
        os.path.join(config.res_dir, config.experiment_name, f"{name}.pth.tar"),
    )


def load_model(config, model, train_out_layer=True):
    # load pre-trained checkpoints, but only of matching weigths

    pretrained_dict = torch.load(config.trained_checkp, map_location=config.device)["state_dict_G"]
    model_dict      = model.netG.state_dict()

    not_str = "" if pretrained_dict.keys() == model_dict.keys() else "not "
    print(f'The new and the (pre-)trained model architectures are {not_str}identical.\n')

    try:# try loading checkpoint strictly, all weights must match
        # (this is satisfied e.g. when resuming training)

        if train_out_layer: raise NotImplementedError # move to 'except' case
        model.netG.load_state_dict(pretrained_dict, strict=True)
        freeze_layers(model.netG, grad=True)    # set all weights to trainable, no need to freeze
        model.frozen, freeze_these = False, []  # ... as all weights match appropriately
    except: # if some weights don't match (e.g. when loading from pre-trained U-Net), then only load the compatible subset ...
        #     ... freeze compatible weights and make the incompatibel weights trainable

        # check for size mismatch and exclude layers whose dimensions mismatch (they won't be loaded)
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.netG.load_state_dict(model_dict, strict=False)

        # freeze pretrained weights
        model.frozen = True
        freeze_layers(model.netG, grad=True) # set all weights to trainable, except final ...
        if train_out_layer:
            # freeze all but last layer
            all_but_last = {k:v for k, v in pretrained_dict.items() if 'out_conv.conv.conv.0' not in k}
            freeze_layers(model.netG, apply_to=all_but_last, grad=False)
            freeze_these = list(all_but_last.keys())
        else: # freeze all pre-trained layers, without exceptions
            freeze_layers(model.netG, apply_to=pretrained_dict, grad=False)
            freeze_these = list(pretrained_dict.keys())
    train_these = [train_layer for train_layer in list(model_dict.keys()) if train_layer not in freeze_these]
    print(f'\nFroze these layers: {freeze_these}')
    print(f'\nTrain these layers: {train_these}')

    if config.resume_from:
        resume_at = int(config.trained_checkp.split('.pth.tar')[0].split('_')[-1])
        print(f'\nResuming training at epoch {resume_at+1}/{config.epochs}, loading optimizers and schedulers')
        # if continuing training, then also load states of previous runs' optimizers and schedulers
        # ---else, we start optimizing from scratch but with the model parameters loaded above
        optimizer_G_dict = torch.load(config.trained_checkp, map_location=config.device)["optimizer_G"]
        model.optimizer_G.load_state_dict(optimizer_G_dict)

        scheduler_G_dict = torch.load(config.trained_checkp, map_location=config.device)["scheduler_G"]
        model.scheduler_G.load_state_dict(scheduler_G_dict)

    # no return value, models are passed by reference


# function to load checkpoints of individual and ensemble models
# (this is used for training and testing scripts)
def load_checkpoint(config, checkp_dir, model, name, path=None):
    composed_path = os.path.join(checkp_dir, config.experiment_name, f"{name}.pth.tar")
    chckp_path    = path if path is not None else composed_path
    print(f'Loading checkpoint {chckp_path}')
    checkpoint = torch.load(chckp_path, map_location=config.device)["state_dict"]

    try: # try loading checkpoint strictly, all weights & their names must match
        model.load_state_dict(checkpoint, strict=True)
    except:
        # rename keys
        #   in_block1 -> in_block0, out_block1 -> out_block0
        checkpoint_renamed = dict()
        for key, val in checkpoint.items():
            if 'in_block' in key or 'out_block' in key:
                strs    = key.split('.')
                strs[1] = strs[1][:-1] + str(int(strs[1][-1])-1)
                strs[1] = '.'.join([strs[1][:-1], strs[1][-1]])
                key     = '.'.join(strs)
            checkpoint_renamed[key] = val
        model.load_state_dict(checkpoint_renamed, strict=False)

def freeze_layers(net, apply_to=None, grad=False):
    if net is not None:
        for k, v in net.named_parameters():
            # check if layer is supposed to be frozen
            if hasattr(v, 'requires_grad') and v.dtype != torch.int64:
                if apply_to is not None:
                    # flip
                    if k in apply_to.keys() and v.size() == apply_to[k].size():
                        v.requires_grad_(grad)
                else: # otherwise apply indiscriminately to all layers
                    v.requires_grad_(grad)
