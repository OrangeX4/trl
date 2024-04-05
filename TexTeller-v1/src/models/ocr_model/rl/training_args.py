CONFIG = {
    "seed": 42,  # Random seed for reproducibility
    # "learning_rate": 5e-5,                 # Learning rate
    # "num_train_epochs": 10,                # Total number of training epochs
    # "per_device_train_batch_size": 4,      # Batch size per GPU for training
    # "per_device_eval_batch_size": 8,       # Batch size per GPU for evaluation
    "pretrained_model": "/home/orangex4/.cache/huggingface/hub/models--OleehyO--TexTeller-Backup/snapshots/4e06f3f0efa19c72a6702b7a7c88c185fb613d44",  # Pretrained model to use
    "ppo_config": {
        "mini_batch_size": 8,
        "batch_size": 8,
        "log_with": "tensorboard",
    },
    "generate_config": {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "max_new_tokens": 256,
    },
    "overwrite_output_dir": False,  # If the output directory exists, do not delete its content
    "save_strategy": "steps",  # Strategy to save checkpoints
    "save_steps": 50,  # Interval of steps to save checkpoints, can be int or a float (0~1), when float it represents the ratio of total training steps (e.g., can set to 1.0 / 2000)
    "save_total_limit": 5,  # Maximum number of models to save. The oldest models will be deleted if this number is exceeded
    "logging_strategy": "steps",  # Log every certain number of steps
    "logging_steps": 500,  # Number of steps between each log
    "logging_nan_inf_filter": False,  # Record logs for loss=nan or inf
}
