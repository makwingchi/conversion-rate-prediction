def get_configurations():
    return {
        "runner": {
            "seed": 666,
            "device": "cpu",
            "model_save_path": "./output_dnn_queue",
            "train_data_path": "../data/data205411/2023-cvr-contest-data/train_data",
            "test_data_path": "../data/data204194/test_data",
            "pipe_command": "python3 ./readers/single_task_reader.py",
            "train_batch_size": 512,
            "train_epochs": 1,
            "infer_batch_size": 512,
            "infer_start_epoch": 0,
            "infer_end_epoch": 1,
            "thread_num": 1,
            "model_type": "baseline"
        },
        "optimizer": {
            "learning_rate": 0.001
        },
        "models": {
            "baseline": {
                "sparse_feature_dim": 9,
                "sparse_feature_number": 88000000,
                "sparse_inputs_slots": 28,
                "fc_sizes": [512, 256, 128, 32]
            }
        }

    }
