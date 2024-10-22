# map arg string of written list to list
def str2list(config, list_args):
    for k, v in vars(config).items():
        if k in list_args and v is not None and isinstance(v, str):
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))
    return config


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)