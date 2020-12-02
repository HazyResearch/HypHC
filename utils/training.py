"""Training utils."""

import argparse
import hashlib
import os


def str2bool(v):
    """Converts string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_flags_from_config(parser, config_dict):
    """Adds a flag (and default value) to an ArgumentParser for each parameter in a config."""

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            else:
                parser.add_argument(f"--{param}", type=OrNone(default), default=default)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser


def hash_dict(values):
    """Hash of dict key, value pairs."""
    m = hashlib.sha256()
    keys = sorted(list(values.keys()))
    for k in keys:
        if k != "seed":
            m.update(str(values[k]).encode('utf-8'))
    return m.hexdigest()


def get_savedir(args):
    """Hash of args used for training."""
    dir_hash = hash_dict(args.__dict__)
    save_dir = os.path.join(os.environ["SAVEPATH"], args.dataset, dir_hash)
    return save_dir
