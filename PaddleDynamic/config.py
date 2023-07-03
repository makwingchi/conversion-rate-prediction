def get_configurations():
    config = {
        "runner": {
            "seed": 666,
            "device": "gpu",
            "model_save_path": "./output_t3",
            "train_data_path": "../work_data/data205411/2023-cvr-contest-data/train_t3",
            "test_data_path": "../work_data/data204194/test_t3",
            "batch_size": 512,
            "train_epochs": 1,
            "thread_num": 1,
            "print_interval": 50,
            "max_len": 20,
            "model_type": "nfm",
            "task_type": "single",
            "conv_type": "3",
            "is_infer": True,
            "neg_coef": 4
        },
        "optimizer": {
            "learning_rate": 0.001
        },
        "models": {
            "common": {
                "sparse_feature_dim": 11,
                "sparse_feature_number": 88000000,
                "num_fields": 26,
                "fc_sizes": [512, 256, 128, 32],
                "activate": "relu"
            },
            "deepandcross": {
                "num_crosses": 3
            },
            "pnn": {
                "num_matrices": 8
            },
            "mmoe": {
                "expert_num": 3,
                "expert_size": 256,
                "tower_size": 128,
                "gate_num": 3
            },
            "dlrm": {
                "self_interaction": True
            },
            "afm": {
                "attention_dim": 8
            },
            "naive_attention": {
                "qkv_dim": 9
            }
        }
    }

    config["runner"]["infer_start_epoch"] = config["runner"]["train_epochs"] - 1
    config["runner"]["infer_end_epoch"] = config["runner"]["train_epochs"]

    return config
