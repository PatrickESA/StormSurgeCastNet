import os
import sys
import json
import pprint
import argparse
import numpy as np
import xarray as xr
from warnings import warn

import shapely
import pandas as pd
import geopandas as gpd

os.environ['WANDB_MODE'] = 'disabled'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'

from tqdm import tqdm
from parse_args import create_parser
from scipy.interpolate import interp1d

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import dask
dask.config.set(scheduler='synchronous')

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dirname))

from util import utils, losses
from util.dataLoader import coastalLoader
from util.model_utils import get_model, load_checkpoint
from train import iterate, save_results, prepare_data, prepare_output, seed_packages, seed_worker

parser = create_parser(mode='test')
test_config = parser.parse_args()

# grab the PID so we can look it up in the logged config for server-side process management
test_config.pid = os.getpid()

# load previous config from training directories

# if no custom path to config file is passed, try fetching config file at default location
conf_path = os.path.join(dirname, test_config.weight_folder, test_config.experiment_name, "conf.json") if not test_config.load_config else test_config.load_config
if os.path.isfile(conf_path):
    with open(conf_path) as file:
        model_config = json.loads(file.read())
        t_args = argparse.Namespace()
        # do not overwrite the following flags by their respective values in the config file
        no_overwrite = ['root', 'pid', 'device', 'resume_at', 'trained_checkp', 'res_dir', 'weight_folder', 'num_workers', 
        'max_samples_count', 'batch_size', 'display_step', 'export_every', 'input_t', 'lead_time', 'eval_gtsm_pred']
        conf_dict = {key:val for key,val in model_config.items() if key not in no_overwrite}
        for key, val in vars(test_config).items(): 
            if key in no_overwrite: conf_dict[key] = val
        t_args.__dict__.update(conf_dict)
        config = parser.parse_args(namespace=t_args)
else: config = test_config # otherwise, keep passed flags without any overwriting
config = utils.str2list(config, ["encoder_widths", "decoder_widths", "out_conv"])

# softly apply changes to parsed arguments, which would otherwise raise errors later on
if config.model in ['lstm', 'conv_lstm']:
    if not config.hyperlocal: 
        warn('A local method was selected together with densification mode. \
            Changing to the hyperlocal experimental.')
        config.hyperlocal = True

experime_dir = os.path.join(config.res_dir, config.experiment_name)
if not os.path.exists(experime_dir): os.makedirs(experime_dir)
with open(os.path.join(experime_dir, "conf.json"), "w") as file:
    file.write(json.dumps(vars(config), indent=4))

# seed everything
seed_packages(config.rdm_seed)
# seed generators for train & val/test dataloaders
f, g = torch.Generator(), torch.Generator()
f.manual_seed(config.rdm_seed + 0)  # note:  this may get re-seeded each epoch
g.manual_seed(config.rdm_seed)      #        keep this one fixed


if __name__ == "__main__": pprint.pprint(config)

models = ['utae', 'metnet3', 'lstm', 'conv_lstm']
# note: in addition, the selected model may also be 
#       'gtsm', 'extrapolate_gtsm, 'inavg', 'seasonal', 'extrapolate_gesla'

def main(config):
    device = torch.device(config.device)
    prepare_output(config)

    if config.model in models:
        only_series = False
        model = get_model(config)
        model = model.to(device)
        model.eval()
        config.N_params = utils.get_ntrainparams(model)
        print(f"TOTAL TRAINABLE PARAMETERS: {config.N_params}\n")
        print(model)

        # Load weights
        ckpt_n = f'_epoch_{config.resume_at}' if config.resume_at > 0 else ''
        load_checkpoint(config, config.weight_folder, model, f"model{ckpt_n}")
    elif config.model in ['gtsm', 'extrapolate_gtsm']:
        only_series = False
    elif config.model in ['inavg', 'seasonal', 'extrapolate_gesla']:
        only_series = True
    else: raise NotImplementedError

    cozy_printing = ['densification', 'hyperlocal'] 
    print(f'Testing {config.model} in {cozy_printing[config.hyperlocal]} evaluation mode.')
    
    root          = os.path.expanduser(config.root)
    stats_file    = os.path.join(root, 'aux', 'stats.npy')
    splits_file   = os.path.join(root, 'aux', 'splits_ids.npy')
    ibtracs_file  = os.path.join(root, 'aux', 'stats_ibtracs.npy')

    stats_data    = None if not os.path.isfile(stats_file) else np.load(stats_file, allow_pickle='TRUE').item()
    splits_ids    = None if not os.path.isfile(splits_file) else np.load(splits_file, allow_pickle='TRUE').item()
    stats_ibtracs = None if not os.path.isfile(ibtracs_file) else np.load(ibtracs_file, allow_pickle='TRUE').item()
    
    dt_test       = coastalLoader(root, split='test', hyperlocal=config.hyperlocal, splits_ids=splits_ids, stats=stats_data, stats_ibtracs=stats_ibtracs, input_len=config.input_t, drop_in=0.0, context_window=config.context, res=config.res, lead_time=config.lead_time, center_gauge=config.center_gauge, only_series=only_series, no_gesla_context=config.no_gesla_context, seed=2) 
    sub_dt_test   = torch.utils.data.Subset(dt_test, range(0, min(config.max_samples_count, len(dt_test))))
    test_loader   = torch.utils.data.DataLoader(sub_dt_test, 
                                                batch_size=config.batch_size, 
                                                shuffle=False, 
                                                worker_init_fn=seed_worker, 
                                                generator=g,
                                                num_workers=config.num_workers)

    print('Loading GESLA dataset into memory')
    ncGESLA = xr.open_dataset(filename_or_obj=os.path.join(root, 'combined_gesla_surge.nc'), engine='netcdf4').load()
    train_points  = [tuple(point) for pdx, point in enumerate(zip(ncGESLA.longitude.values, ncGESLA.latitude.values)) if ncGESLA.isel(station=pdx).station.values in dt_test.splits_ids['train']]
    gesla_points  = gpd.GeoDataFrame(geometry=shapely.points(coords=train_points))

    # Inference
    print("Testing . . .")
    pred, targ, dist, context_n, coords = [], [], [], [], []
    for batch in tqdm(test_loader):
        if config.model in models:
            x, y, in_m, dates, lead = prepare_data(batch, device, config)
            inputs = {'A': x, 'B': y, 'dates': dates, 'masks': in_m, 'lead': lead}
            with torch.no_grad():
                # compute mean predictions
                model.set_input(inputs)
                model.forward()
                model.get_loss_G()
                out = model.fake_B
            if config.use_series_target:
                preds = out
            else:
                if config.eval_gtsm_pred:
                    out = out[:, :, -1, ...]
                else:
                    out = out[:, :, 0, ...]
                validity_mask = ~np.isnan(batch['target']['sparse']).bool()
                preds = out[validity_mask]

        if config.model == 'gtsm': # forced by future ERA5 reanalysis
            # target data is [B x 1 x H x W]
            validity_mask = ~np.isnan(batch['target']['sparse']).bool()
            sparse_values = batch['target']['sparse'][validity_mask]
            out           = batch['target']['gtsm_out_unmasked']
            preds         = out[validity_mask]

        if config.model == "extrapolate_gtsm": # extrapolating from input time series
            validity_mask = ~np.isnan(batch['target']['sparse']).bool()
            sparse_values = batch['target']['sparse'][validity_mask]
            t_times_mask  = validity_mask[:,None,...].expand(-1,config.input_t,-1,-1,-1)

            preds = []
            for bdx, bitem in enumerate(batch['input']['gtsm'].squeeze()): # iterate over current batch items
                try:
                    interpolator = interp1d(np.arange(config.input_t), bitem[t_times_mask[bdx,:,0,...]], kind='linear', axis=-1, fill_value="extrapolate") # bitem is T x H x W
                    inter = interpolator(np.arange(config.input_t + config.lead_time))[-1]
                    preds.append(inter)
                except: continue 
            preds = torch.tensor(preds).float()

        impute = False
        if config.model == "extrapolate_gesla":
            validity_mask = ~np.isnan(batch['target']['series']).bool()
            preds         = np.array([interp1d(np.arange(config.input_t), bitem, kind='linear', axis=-1, fill_value="extrapolate")(np.arange(config.input_t + config.lead_time))[-1] for bitem in batch['input']['series'].squeeze()])
        if config.model == 'seasonal':
            validity_mask  = ~np.isnan(batch['target']['series']).bool()
            dates          = np.array([[pd.to_datetime(batch_item).values[-1] + np.timedelta64(config.lead_time,'h') for batch_item in batch['input']['dates2int']]])[0]
            prevYearsMonth = [ncGESLA.sel(station=batch['target']['id'][ddx], date_time=ncGESLA.date_time.dt.month.isin([int(str(dates[0])[6])])).sel(date_time=slice(ncGESLA.date_time[0], ditem)) for ddx, ditem in enumerate(dates)]
            preds          = np.array([np.nanmean(periods.sea_level.values) for periods in prevYearsMonth])
            impute         = np.isnan(preds).sum() > 0 # NaNs in prediction can appear depending on record length of current GESLA gauge, in this case impute missing values via 'inavg' baseline

        if config.model == 'inavg' or impute:
            validity_mask = ~np.isnan(batch['target']['series']).bool()
            dates_a   = np.array([[pd.to_datetime(batch_item).values[0] for batch_item in batch['input']['dates2int']]])[0]
            dates_b   = np.array([[pd.to_datetime(batch_item).values[-1] for batch_item in batch['input']['dates2int']]])[0]
            intervals = [ncGESLA.sel(date_time=slice(dates_a[bdx], dates_b[bdx]), station=station) for bdx, station in enumerate(batch['target']['id'])]
            if impute:
                # impute missing values of another method's preceding predictions
                preds[np.isnan(preds)] = torch.Tensor([np.nanmean(gesla.sea_level.values) for gesla in intervals])[np.isnan(preds)]
            else:
                preds = torch.Tensor([np.nanmean(gesla.sea_level.values) for gesla in intervals])

        if only_series or config.use_series_target:
            sparse_values = batch['target']['series'].squeeze()
            lon, lat      = batch['input']['lon'].numpy(), batch['input']['lat'].numpy()
        else:
            sparse_values = batch['target']['sparse'][validity_mask].squeeze()
            lon, lat      = batch['target']['lon_gauge'].numpy(), batch['target']['lat_gauge'].numpy()

        # for each target sample, collect its distance to the closest train split gauge
        target_p      = [tuple(point) for pdx, point in enumerate(zip(lon, lat)) if pdx < torch.numel(sparse_values)]
        target_points = gpd.GeoDataFrame(geometry=shapely.points(coords=target_p))
        dist_mat      = gesla_points.geometry.apply(lambda g: target_points.distance(g))
        closest_idx   = np.argmin(dist_mat, axis=0)
        closest_dist  = dist_mat.iloc[closest_idx, :].values[np.eye(len(target_points), dtype=bool)]
        
        coords += target_p
        targ   += sparse_values.tolist()
        pred   += preds.tolist()
        dist   += closest_dist.tolist()

        if 'valid_mask' in batch['input']:
            context_n += batch['input']['valid_mask'].sum(-1).sum(-1)[...,-1].mean(-1).int()[:torch.numel(sparse_values)].tolist()

    # compute summary statistics across the entire split
    pred, targ = torch.tensor(pred).squeeze(), torch.tensor(targ).squeeze()
    mean_mae = torch.nanmean(torch.nn.functional.l1_loss(targ, pred, reduction='none'))
    mean_mse = torch.nanmean(torch.nn.functional.mse_loss(targ, pred, reduction='none'))
    std_mae = np.nanstd(torch.nn.functional.l1_loss(targ, pred, reduction='none'))
    std_mse = np.nanstd(torch.nn.functional.mse_loss(targ, pred, reduction='none'))
    nnse    = losses.nnse(np.array(targ), np.array(pred))
    print(f'Standardized {config.model}: MAE {mean_mae} ({std_mae}), MSE {mean_mse} ({std_mse}), NNSE {nnse}')

    m_targ  = dt_test.stats['std']['GESLA'] * targ + dt_test.stats['mean']['GESLA']
    m_pred  = dt_test.stats['std']['GTSM'] * pred + dt_test.stats['mean']['GTSM']
    mean_mae = torch.nanmean(torch.nn.functional.l1_loss(m_targ, m_pred, reduction='none'))
    mean_mse = torch.nanmean(torch.nn.functional.mse_loss(m_targ, m_pred, reduction='none'))
    std_mae = np.nanstd(torch.nn.functional.l1_loss(m_targ, m_pred, reduction='none'))
    std_mse = np.nanstd(torch.nn.functional.mse_loss(m_targ, m_pred, reduction='none'))
    m_nnse  = losses.nnse(np.array(m_targ), np.array(m_pred))
    print(f'm-units {config.model}: MAE {mean_mae} ({std_mae}), MSE {mean_mse} ({std_mse}), NNSE {m_nnse}')
    print(f'Statistics: {dt_test.stats}')


if __name__ == "__main__":
    main(config)
    exit()
