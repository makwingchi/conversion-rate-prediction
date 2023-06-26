def get_configurations():
    return {
        "runner": {
            "seed": 666,
            "device": "gpu",
            "model_save_path": "./output_dnn",
            "train_data_path": "../data/data205411/2023-cvr-contest-data/train_data",
            "test_data_path": "../data/data204194/test_data",
            "batch_size": 512,
            "train_epochs": 1,
            "infer_start_epoch": 0,
            "infer_end_epoch": 1,
            "thread_num": 1,
            "print_interval": 50,
            "max_len": 21,
            "model_type": "baseline",
            "task_type": "single"
        },
        "optimizer": {
            "learning_rate": 0.001
        },
        "models": {
            "baseline": {
                "sparse_feature_dim": 9,
                "sparse_feature_number": 88000000,
                "num_fields": 26,
                "fc_sizes": [512, 256, 128, 32]
            },
            "deepcrossing": {
                "sparse_feature_dim": 9,
                "sparse_feature_number": 88000000,
                "num_fields": 26,
                "fc_sizes": [512, 256, 128, 32]
            },
            "wideanddeep": {
                "sparse_feature_dim": 9,
                "sparse_feature_number": 88000000,
                "num_fields": 26,
                "fc_sizes": [512, 256, 128, 32]
            },
            "deepandcross": {
                "sparse_feature_dim": 9,
                "sparse_feature_number": 88000000,
                "num_fields": 26,
                "fc_sizes": [512, 256, 128, 32],
                "num_crosses": 3
            },
            "fnn": {
                "sparse_feature_dim": 9,
                "sparse_feature_number": 88000000,
                "num_fields": 26,
                "fc_sizes": [512, 256, 128, 32]
            },
            "pnn": {
                "sparse_feature_dim": 9,
                "sparse_feature_number": 88000000,
                "num_fields": 26,
                "fc_sizes": [512, 256, 128, 32],
                "num_matrices": 8
            }
        }
    }
