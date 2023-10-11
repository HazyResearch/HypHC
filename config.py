"""Configuration parameters."""

config_args = {
    # training
    "seed": 1234,
    "epochs": 50,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "eval_every": 10,
    "patience": 20,
    "optimizer": "RAdam",
    "save": 1,
    "fast_decoding": 1,
    "num_samples": -1,
    "large_dataset": 0,
    
    # model
    "dtype": "double",
    "rank": 2,
    "temperature": 0.01,
    "init_size": 1e-3,
    "anneal_every": 20,
    "anneal_factor": 1.0,
    "max_scale": 1 - 1e-3,

    # dataset
    "dataset": "zoo",
    "data_points": 101,
}
