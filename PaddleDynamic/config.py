def get_configurations(conv_type, purpose, model_type, is_infer):
    config = {
        "runner": {
            "seed": 666,
            "device": "gpu",
            "batch_size": 512,
            "train_epochs": 1,
            "thread_num": 1,
            "print_interval": 50,
            "max_len": 20,
            "task_type": "single",
            "neg_coef": 3,
            "is_shuffle": False
        },
        "optimizer": {
            "learning_rate": 0.001
        },
        "models": {
            "common": {
                "sparse_feature_dim": 11,
                "sparse_feature_number": 88000000,
                "num_fields": 26,
                "fc_sizes": [256, 128, 32],
                "activate": "relu"
            },
            "deepandcross": {
                "num_crosses": 3,
                "mlp_sizes": [8]
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
            "sharebottom": {
                "num_tasks": 3
            },
            "dlrm": {
                "self_interaction": True
            },
            "afm": {
                "attention_dim": 8
            },
            "autoint": {
                "attn_layer_sizes": [32, 32, 32],
                "head_num": 4
            },
            "naive_attention": {
                "qkv_dim": 9
            },
            "deepcrossingattn": {
                "attention_sizes": [32, 16, 8]
            },
            "deepandcrossv2": {
                "num_crosses": 3,
                "is_Stacked": False,
                "use_low_rank_mixture": False,
                "low_rank": 32
            },
            "deepandcrossv6": {
                "excitation": "relu"
            }
        }
    }

    config["runner"]["model_type"] = model_type
    config["runner"]["conv_type"] = str(conv_type)
    config["runner"]["model_save_path"] = "./output_t" + str(conv_type)
    config["runner"]["is_infer"] = is_infer

    if purpose.lower() == "test":
        config["runner"]["train_data_path"] = "../work_data/data205411/2023-cvr-contest-data/train_t" + str(conv_type)
        config["runner"]["test_data_path"] = "../work_data/data204194/test_t" + str(conv_type)
    elif purpose.lower() == "validation":
        config["runner"]["train_data_path"] = "../work_data/data205411/2023-cvr-contest-data/first28_t" + str(conv_type)
        config["runner"]["test_data_path"] = "../work_data/data204194/last2_t" + str(conv_type)

    if config["runner"]["task_type"] == "multi":
        config["runner"]["train_data_path"] = "../work_data/data205411/2023-cvr-contest-data/train_data"
        config["runner"]["test_data_path"] = "../work_data/data204194/test_data"

    config["runner"]["infer_start_epoch"] = config["runner"]["train_epochs"] - 1
    config["runner"]["infer_end_epoch"] = config["runner"]["train_epochs"]

    return config
