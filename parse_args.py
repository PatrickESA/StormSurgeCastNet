import os
import argparse

# input:
#   2 layers of sparse in-situ measurements and dense GTSM simulations
#   3 layers of ERA5 data
#   1 layer of valid/invalid mask for indicating location of sparse observations
IN_BANDS  = 2 + 3 + 1

# output:
#   1 layer of densified in-situ measurements
#   1 layer of coarse GTSM predictions (just for supervision purposes)
OUT_BANDS = 1 + 1


def str2bool(val):
    if val in {None, 'None'}: return None
    elif isinstance(val, bool): return val
    elif val.lower() in {'true', 't', 'yes', 'y', '1'}: return True
    elif val.lower() in {'false', 'f', 'no', 'n', '0'}: return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')
    

def create_parser(mode='train'):
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument(
        "--model",
        default='utae',
        type=str,
        help="Type of architecture to use. Can be one of: (lstm/conv_lstm/utae/metnet3)",
    )
    parser.add_argument("--experiment_name", default='dbg', help="Name of the current experiment",)

    # fast switching between default arguments, depending on train versus test mode
    if mode=='train':
        parser.add_argument("--res_dir", default="./results", help="Path to where the results are stored, e.g. ./results for training or ./inference for testing",)
        parser.add_argument("--export_every", default=-1, type=int, help="Interval (in items) of exporting data at validation or test time. Set -1 to disable")
        parser.add_argument("--resume_at", default=0, type=int, help="Epoch to resume training from (may re-weight --lr in the optimizer) or epoch to load checkpoint from at test time")
    elif mode=='test':
        parser.add_argument("--res_dir", default="./inference", type=str, help="Path to directory where results are written.")
        parser.add_argument("--export_every", default=1, type=int, help="Interval (in items) of exporting data at validation or test time. Set -1 to disable")
        parser.add_argument("--resume_at", default=0, type=int, help="Epoch to load checkpoint from and run testing with (use -1 for best on validation split)")

    # set-up parameters
    parser.add_argument("--num_workers", default=12, type=int, help="Number of data loading workers")
    parser.add_argument("--rdm_seed", default=1, type=int, help="Random seed")
    parser.add_argument("--device",default="cuda",type=str,help="Name of device to use for tensor computations (cuda/cpu)",)
    parser.add_argument("--display_step", default=10, type=int, help="Interval in batches between display of training metrics",)

    parser.add_argument("--in_dim", default=IN_BANDS, type=int, help="dimension of the input features, used for defining models")
    parser.add_argument("--out_conv", default=f"[{OUT_BANDS}]", help="output CONV, note: if inserting another layer then consider treating normalizations separately")
    parser.add_argument("--mean_nonLinearity", dest="mean_nonLinearity", action="store_true", help="whether to apply a sigmoidal output nonlinearity to the mean prediction") 

    parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str, help="e.g. [64,64,64,128] for U-TAE")
    parser.add_argument("--decoder_widths", default="[64,64,64,128]", type=str, help="e.g. [64,64,64,128] for U-TAE")
    parser.add_argument("--agg_mode", default="att_group", type=str, help="type of temporal aggregation in L-TAE module")
    parser.add_argument("--encoder_norm", default="group", type=str, help="e.g. 'group' (when using many channels) or 'instance' (for few channels)")
    parser.add_argument("--decoder_norm", default="batch", type=str, help="e.g. 'group' (when using many channels) or 'instance' (for few channels)")
    parser.add_argument("--padding_mode", default="reflect", type=str)
    parser.add_argument("--pad_value", default=0, type=float)

    # attention-specific parameters
    parser.add_argument("--n_head", default=16, type=int, help="default value of 16, 4 for debugging")
    parser.add_argument("--d_model", default=256, type=int, help="layers in L-TAE, default value of 256")
    parser.add_argument("--positional_encoding", dest="positional_encoding", action="store_false", help="whether to use positional encoding or not") 
    parser.add_argument("--d_k", default=4, type=int)

    # training parameters
    parser.add_argument("--loss", default="weighted-l1", type=str, help="Loss to utilize [weighted-l1|l1|l2].")
    parser.add_argument("--resume_from", dest="resume_from", action="store_true", help="resume training acc. to JSON in --experiment_name and *.pth chckp in --trained_checkp")
    parser.add_argument("--unfreeze_after", default=0, type=int, help="When to unfreeze ALL weights for training")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--chunk_size", type=int, help="Size of vmap batches, this can be adjusted to accommodate for additional memory needs")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate, e.g. 0.001")
    parser.add_argument("--gamma", default=0.9, type=float, help="Learning rate decay parameter for scheduler")
    parser.add_argument("--val_every", default=1, type=int, help="Interval in epochs between two validation steps.")
    parser.add_argument("--val_after", default=0, type=int, help="Do validation only after that many epochs.")

    # parameters regarding lead time and lead time conditioning
    parser.add_argument("--lead_time", default=8, type=int, help="Lead time for which to predict target values") # note: keep out of smoothing window?
    parser.add_argument("--max_lead_times", default=16, type=int, help="Maximum lead time randomly sampled for augmentation at train time")
    parser.add_argument("--film", dest="film", action="store_false", help="whether to use FiLM conditioning or not")
    parser.add_argument("--film_latent", default=32, type=int, help="latent space of the FiLM embedding")
    parser.add_argument('--cond_norm_affine', type=str2bool, nargs='?', const=None, help='for UTAE: whether to (not) use affine norm layers or use default behavior (None)')

    # flags specific to surge forecasting dataset
    parser.add_argument("--max_samples_count", default=int(1e9), type=int, help="count of data (sub-)samples to take") # int(1e9), debug @ 4
    parser.add_argument("--max_samples_frac", default=1.0, type=float, help="fraction of data (sub-)samples to take")
    parser.add_argument("--vary_samples", dest="vary_samples", action="store_false", help="whether to sample different time points across epochs or not")
    parser.add_argument("--center_gauge", action="store_true", help="center gauge in sampled roi; defaults to randomly sampling nearby areas")
    parser.add_argument("--no_gesla_context", action="store_true", help="only use selected gauge in gesla raster")
    parser.add_argument("--weighting", default=1/100.0, type=float, help="The weighting for combining sparse versus coarse losses")
    parser.add_argument("--drop_data", default=0.25, type=float, help="The rate of dropping out sparse in situ data from the input time series")
    parser.add_argument("--input_t", default=12, type=int, help="number of input time points to sample") # 12, debug @ 2
    parser.add_argument("--root", default=os.path.expanduser('~/Data'), type=str, help="path to your copy of the dataset")
    parser.add_argument("--trained_checkp", default="", type=str, help="Path to loading a pre-trained network *.pth file, rather than initializing weights randomly")
    parser.add_argument("--res", default=0.025, type=float, help="The raster resolution in terms of degrees (0.1 deg is circa 10 km)")
    parser.add_argument("--context", default=128, type=int, help="Raster sampled distance from center, i.e. half the height & width")
    parser.add_argument("--era5", dest="era5", action="store_false", help="whether to use ERA5 input or not")
    parser.add_argument("--gtsm", dest="gtsm", action="store_false", help="whether to use GTSM input or not")
    parser.add_argument("--hyperlocal", dest="hyperlocal", action="store_true", help="whether to evaluate in hyperlocal or densifying mode") 
    parser.add_argument("--use_series_input", action="store_true", help="expect model to accept series inputs (e.g. LSTM)")
    parser.add_argument("--use_series_target", action="store_true", help="expect model to produce series targets (e.g. LSTM)")
    parser.add_argument("--eval_gtsm_pred", action="store_true", help="evaluating on the model's coarse GTSM forecast, instead of its densified prediction")

    # flags specific for testing
    parser.add_argument("--weight_folder", type=str, default="./results", help="Path to the main folder containing the pre-trained weights")
    parser.add_argument("--load_config", default='', type=str, help="path of conf.json file to load")
    
    return parser
