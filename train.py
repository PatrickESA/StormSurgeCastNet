import os
import sys
import time
import json
import random
import numpy as np
from tqdm import tqdm
from warnings import warn
from matplotlib import pyplot as plt

os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'

import torch
sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

import dask
dask.config.set(scheduler='synchronous')

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)

from parse_args import create_parser

import util.meter as meter
from util import utils, losses
from util.weight_init import weight_init
from util.dataLoader import coastalLoader
from util.metrics import avg_img_metrics
from util.model_utils import get_model, save_model, freeze_layers, load_model, load_checkpoint

parser   = create_parser(mode='train')
config   = utils.str2list(parser.parse_args(), list_args=["encoder_widths", "decoder_widths", "out_conv"])

# softly apply changes to parsed arguments, which would otherwise raise errors later on
if config.model in ['lstm', 'conv_lstm']:
    config.loss, config.drop_data = 'l1', 0.0
    if not config.hyperlocal: 
        warn('A local method was selected together with densification mode. \
            Changing to the hyperlocal experimental.')
        config.hyperlocal = True
    if config.film: 
        warn('A method without lead time conditioning was selected. \
            Changing to no lead time conditioning.')
        config.film = False
    if not (config.use_series_target and config.center_gauge):
            warn('A 1D method in combination with 2D target data was selected. Changing to 1D data.')
            config.use_series_target, config.center_gauge, config.context = True, True, 2


# predict:
#   1 layer of densified in-situ measurements
#   1 layer of coarse GTSM predictions (just for supervision purposes)
OUT_BANDS = config.out_conv[0] # 1 + 1

# resume at a specified epoch and update optimizer accordingly
if config.resume_at >= 0:
    config.lr = config.lr * config.gamma**config.resume_at


# fix all RNG seeds
def seed_packages(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# seed everything
seed_packages(config.rdm_seed)
# seed generators for train & val/test dataloaders
f, g = torch.Generator(), torch.Generator()
f.manual_seed(config.rdm_seed + 0)  # note:  this may get re-seeded each epoch
g.manual_seed(config.rdm_seed)      #        keep this one fixed


def iterate(model, data_loader, config, mode="train", epoch=None, device=None):
    if len(data_loader) == 0: raise ValueError("Received data loader with zero samples!")

    loss_meter = meter.AverageValueMeter()
    img_meter  = avg_img_metrics()

    t_start = time.time()
    for i, batch in enumerate(tqdm(data_loader)):
        step = (epoch-1)*len(data_loader)+i

        x, y, in_m, dates, lead = prepare_data(batch, device, config)
        inputs = {'A': x, 'B': y, 'dates': dates, 'masks': in_m, 'lead': lead}


        if mode != "train": # val or test
            with torch.no_grad():
                # compute mean predictions
                model.set_input(inputs)
                model.forward()
                model.get_loss_G()
                out = model.fake_B
                out = out[:, :, :OUT_BANDS, ...]
                batch_size = y.size()[0]
        else: # training
            # compute mean predictions
            model.set_input(inputs)
            model.optimize_parameters() # not using model.forward() directly
            out = model.fake_B.detach().cpu()
            out = out[:, :, :OUT_BANDS, ...]
            
        if mode == "train":
            # periodically log stats
            if step%config.display_step==0:
                out, x, y, in_m = out.cpu(), x.cpu(), y.cpu(), in_m.cpu()
                log_train(config, model, step, x, out, y, in_m)

        # log the loss, computed via model.backward_G() at train time & via model.get_loss_G() at val/test time
        loss_meter.add(model.loss_G.item())
        wandb.log({"train/loss": model.loss_G.item()}, commit=True)

        # after each batch, close any leftover figures
        plt.close('all')

    # --- end of epoch ---

    # after each epoch, log the loss metrics
    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    metrics = {f"{mode}_epoch_time": total_time}

    # log the loss, only computed within model.backward_G() at train time
    metrics[f"{mode}_loss"] = loss_meter.value()[0]

    if mode == "train": # after each epoch, update lr acc. to scheduler
        current_lr = model.optimizer_G.state_dict()['param_groups'][0]['lr']
        wandb.log({"train/lr": current_lr})
        model.scheduler_G.step()
    
    if mode == "test" or mode == "val":
        # log the metrics

        # any loss is currently only computed within model.backward_G() at train time
        loss_dict = {f'{mode}/loss': metrics[f"{mode}_loss"]}
        wandb.log(loss_dict | {key: val for key, val in img_meter.value().items()})
        return metrics, img_meter.value()
    else:
        return metrics

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]

def prepare_output(config):
    os.makedirs(os.path.join(config.res_dir, config.experiment_name), exist_ok=True)

def checkpoint(log, config):
    with open(
        os.path.join(config.res_dir, config.experiment_name, "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)

def save_results(metrics, path, split='test'):
    with open(
        os.path.join(path, f"{split}_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)

import wandb
wandb.login()

run = wandb.init(
    # Set the project where this run will be logged
    project= config.experiment_name, #"storm_surge",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": config.lr,
        "epochs": config.epochs,
        "seed": config.rdm_seed,
    },
)


def log_train(config, model, step, x, out, y, in_m, name=''):
    # logged loss is before rescaling by learning rate
    _, loss = model.criterion, model.loss_G.cpu()
    if name != '': name = f'model_{name}/'

    if not config.use_series_target:
        for bdx in range(x.shape[2]):
            wandb.log({f"train/x.band{bdx}": [wandb.Image(np.nan_to_num(x[0,0,bdx,...]), caption=f"Input image band {bdx}")]}, commit=False)
        for bdx in range(out.shape[2]):
            wandb.log({f"train/out.band{bdx}": [wandb.Image(np.nan_to_num(out[0,0,bdx,...]), caption=f"Prediction band {bdx}")]}, commit=False)
        for bdx in range(y.shape[2]):
            wandb.log({f"train/y.band{bdx}": [wandb.Image(np.nan_to_num(y[0,0,bdx,...]), caption=f"Target image band {bdx}")]}, commit=False)
        wandb.log({f"train/lsm": [wandb.Image(in_m[0,0,...], caption=f"Land Sea Mask")]}, commit=False)

    wandb.log({f'train/{name}total': loss, f'train/{name}{config.loss}': loss}, commit=False)
    # use add_images for batch-wise adding across temporal dimension
    
def prepare_data(batch, device, config):
    batch = recursive_todevice(batch, device)

    # Get main input
    use_series_input = config.use_series_input or ('gtsm' not in batch['input'])
    if use_series_input:
        x = batch['input']['series']
    else:
        in_sparse  = batch['input']['sparse']
        in_valid_m = batch['input']['valid_mask']
        x = torch.cat((in_sparse, in_valid_m), dim=2)
        if config.era5:
            in_era5    = batch['input']['era5']
            x = torch.cat((x, in_era5), dim=2)
        if config.gtsm:
            in_gtsm    = batch['input']['gtsm']
            x = torch.cat((x, in_gtsm), dim=2)
    
    # Get main output
    use_series_target = config.use_series_target or ('gtsm' not in batch['input'])
    if use_series_target:
        y = batch['target']['series']
    else:
        if config.out_conv[-1] > 1: 
            y   = torch.cat((batch['target']['sparse'], batch['target']['gtsm']), dim=1).unsqueeze(1)
        else: y = batch['target']['sparse'][:,:,None,...] # introduce singleton channel dimension

    # Get extra details
    in_m        = batch['input']['ls_mask']
    dates       = batch['input']['td']
    lead        = batch['input']['td_lead']
    # TODO: make use of batch lon/lat

    if torch.isnan(x).sum() > 0:
        print('Encountered NaNs in input data')
        exit()

    return x, y, in_m, dates, lead

# for hyperparameter sweeping, see:
# https://docs.wandb.ai/guides/sweeps
# https://docs.wandb.ai/guides/sweeps/define-sweep-configuration

def main(config):
    prepare_output(config)
    device = torch.device(config.device)

    print('Setting up data loaders.\n')
    root          = os.path.expanduser(config.root)
    stats_file    = os.path.join(root, 'stats.npy')
    splits_file   = os.path.join(root, 'splits_ids.npy')
    ibtracs_file  = os.path.join(root, 'stats_ibtracs.npy')

    stats_data    = None if not os.path.isfile(stats_file) else np.load(stats_file, allow_pickle='TRUE').item()
    splits_ids    = None if not os.path.isfile(splits_file) else np.load(splits_file, allow_pickle='TRUE').item()
    stats_ibtracs = None if not os.path.isfile(ibtracs_file) else np.load(ibtracs_file, allow_pickle='TRUE').item()

    # note: hard-coding in situ input dropout to 0 for validation and test split
    train_lead  = None if config.film else config.lead_time # do not fixate the lead time during training if using FiLM conditioning
    dt_train    = coastalLoader(root, split='train', hyperlocal=True, splits_ids=splits_ids, stats=stats_data, stats_ibtracs=stats_ibtracs, input_len=config.input_t, drop_in=config.drop_data, context_window=config.context, res=config.res, lead_time=train_lead, center_gauge=config.center_gauge, no_gesla_context=config.no_gesla_context)
    dt_val      = coastalLoader(root, split='val', hyperlocal=config.hyperlocal, splits_ids=dt_train.splits_ids, stats=dt_train.stats, stats_ibtracs=dt_train.stats_ibtracs, input_len=config.input_t, drop_in=0.0, context_window=config.context, res=config.res, lead_time=config.lead_time, center_gauge=config.center_gauge, no_gesla_context=config.no_gesla_context, seed=1)
    dt_test     = coastalLoader(root, split='test', hyperlocal=config.hyperlocal, splits_ids=dt_train.splits_ids, stats=dt_train.stats, stats_ibtracs=dt_train.stats_ibtracs, input_len=config.input_t, drop_in=0.0, context_window=config.context, res=config.res, lead_time=config.lead_time, center_gauge=config.center_gauge, no_gesla_context=config.no_gesla_context, seed=2)

    if not os.path.isfile(stats_file): np.save(stats_file, dt_train.stats)
    if not os.path.isfile(ibtracs_file): np.save(ibtracs_file, dt_train.stats_ibtracs)
    if not os.path.isfile(splits_file): np.save(splits_file, dt_train.splits_ids)

    # wrap to allow for subsampling, e.g. for test runs etc
    sub_dt_train    = torch.utils.data.Subset(dt_train, range(0, min(config.max_samples_count, len(dt_train), int(len(dt_train)*config.max_samples_frac))))
    sub_dt_val      = torch.utils.data.Subset(dt_val, range(0, min(config.max_samples_count, len(dt_val), int(len(dt_train)*config.max_samples_frac))))
    sub_dt_test     = torch.utils.data.Subset(dt_test, range(0, min(config.max_samples_count, len(dt_test), int(len(dt_train)*config.max_samples_frac))))

    # instantiate dataloaders, note: worker_init_fn is needed to get reproducible random samples across runs if vary_samples=True
    train_loader = torch.utils.data.DataLoader(
        sub_dt_train,
        batch_size=config.batch_size,
        shuffle=True,
        worker_init_fn=seed_worker, generator=f,
        num_workers=config.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        sub_dt_val,
        batch_size=config.batch_size,
        shuffle=False, # iterate through samples in order
        worker_init_fn=seed_worker, generator=g,
        num_workers=config.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        sub_dt_test,
        batch_size=config.batch_size,
        shuffle=False, # iterate through samples in order
        worker_init_fn=seed_worker, generator=g,
        num_workers=config.num_workers,
    )

    print("Train {}, Val {}, Test {}".format(len(sub_dt_train), len(sub_dt_val), len(sub_dt_test)))

    # model definition
    # (compiled model hangs up in validation step on some systems, retry in the future for pytorch > 2.0)
    model = get_model(config) #torch.compile(get_model(config))

    # set model properties
    model.len_epoch = len(train_loader)
    config.N_params = utils.get_ntrainparams(model)
    model = model.to(device)
    
    # do random weight initialization
    print('\nInitializing weights randomly.')
    model.netG.apply(weight_init)

    if config.trained_checkp and len(config.trained_checkp)>0:
        # load weights from the indicated checkpoint
        print(f'Loading weights from (pre-)trained checkpoint {config.trained_checkp}')
        load_model(config, model, train_out_layer=True)

    with open(os.path.join(config.res_dir, config.experiment_name, "conf.json"), "w") as file:
        file.write(json.dumps(vars(config), indent=4))
    print(f"TOTAL TRAINABLE PARAMETERS: {config.N_params}\n")
    print(model)

    # Optimizer and Loss
    model.criterion = losses.get_loss(config)

    # track best loss, checkpoint at best validation performance
    is_better, best_loss = lambda new, prev: new <= prev, float("inf")

    # Training loop
    trainlog = {}

    # resume training at scheduler's latest epoch, != 0 if --resume_from
    begin_at = config.resume_at if config.resume_at >= 0 else model.scheduler_G.state_dict()['last_epoch']
    for epoch in range(begin_at+1, config.epochs + 1):
        print("\nEPOCH {}/{}".format(epoch, config.epochs))

        # put all networks in training mode again
        model.train()
        model.netG.train()

        # unfreeze all layers after specified epoch
        if epoch>config.unfreeze_after and hasattr(model, 'frozen') and model.frozen:
            print('Unfreezing all network layers')
            model.frozen = False
            freeze_layers(model.netG, grad=True)

        # re-seed train generator for each epoch anew, depending on seed choice plus current epoch number
        #   ~ else, dataloader provides same samples no matter what epoch training starts/resumes from
        #   ~ note: only re-seed train split dataloader (if config.vary_samples), but keep all others consistent
        #   ~ if desiring different runs, then the seeds must at least be config.epochs numbers apart
        if config.vary_samples:
            # condition dataloader samples on current epoch count
            f.manual_seed(config.rdm_seed + epoch)
            train_loader = torch.utils.data.DataLoader(
                            sub_dt_train,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=seed_worker, generator=f,
                            num_workers=config.num_workers,
                            )

        train_metrics = iterate(
                            model,
                            data_loader=train_loader,
                            config=config,
                            mode="train",
                            epoch=epoch,
                            device=device,
                        )

        # do regular validation steps at the end of each training epoch
        if epoch % config.val_every == 0 and epoch > config.val_after:
            print("\nValidation . . . ")

            model.eval()
            model.netG.eval()

            val_metrics, val_img_metrics = iterate(
                                                model,
                                                data_loader=val_loader,
                                                config=config,
                                                mode="val",
                                                epoch=epoch,
                                                device=device,
                                            )
            print(f'Encountered a total of {dt_val.storm_dates} storm dates over {dt_val.storm_count} gauges in val split.')
            dt_val.storm_dates, dt_val.storm_count = 0, 0

            # use the training loss for validation
            print('Using training loss as validation loss')
            if "val_loss" in val_metrics: val_loss = val_metrics["val_loss"]
            else: val_loss = val_metrics['val_loss_ensembleAverage']


            print(f'Validation Loss {val_loss}')
            print(f'validation image metrics: {val_img_metrics}')
            save_results(val_img_metrics, os.path.join(config.res_dir, config.experiment_name), split=f'val_epoch_{epoch}')
            print(f'\nLogged validation epoch {epoch} metrics to path {os.path.join(config.res_dir, config.experiment_name)}')

            # checkpoint best model
            trainlog[epoch] = {**train_metrics, **val_metrics}
            checkpoint(trainlog, config)
            if is_better(val_loss, best_loss):
                best_loss = val_loss
                save_model(config, epoch, model, "model")
        else:
            trainlog[epoch] = {**train_metrics}
            checkpoint(trainlog, config)

        # always checkpoint the current epoch's model
        save_model(config, epoch, model, f"model_epoch_{epoch}")

        print(f'Completed current epoch of experiment {config.experiment_name}.')

    # following training, test on hold-out data
    print("\nTesting model from best epoch . . .")
    try:
        load_checkpoint(config, config.res_dir, model, "model")
    except:
        print('Couldn\'t find best model, defaulting to first epoch\'s model instead.')
        load_checkpoint(config, config.res_dir, model, "model_epoch_1")

    model.eval()
    model.netG.eval()

    test_metrics, test_img_metrics = iterate(
                                        model,
                                        data_loader=test_loader,
                                        config=config,
                                        mode="test",
                                        epoch=1,
                                        device=device,
                                    )
    print(f'Encountered a total of {dt_test.storm_dates} storm dates over {dt_test.storm_count} gauges in test split.')
    dt_test.storm_dates, dt_test.storm_count = 0, 0

    if "test_loss" in test_metrics: test_loss = test_metrics["test_loss"]
    else: test_loss = test_metrics['test_loss_ensembleAverage']
    print(f'Test Loss {test_loss}')
    print(f'\nTest image metrics: {test_img_metrics}')
    save_results(test_img_metrics, os.path.join(config.res_dir, config.experiment_name), split='test')
    print(f'\nLogged test metrics to path {os.path.join(config.res_dir, config.experiment_name)}')

    print(f'Finished running experiment {config.experiment_name}.')

    # close WandB logging
    wandb.finish()

if __name__ == "__main__":
    main(config)
    exit()
