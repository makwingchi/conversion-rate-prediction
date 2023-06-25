import paddle


def dataset_train(epoch_id, dataset, fetch_vars, exe):
    fetch_info = [
        "Epoch {} Var {}".format(epoch_id, var_name) for var_name in fetch_vars
    ]

    fetch_vars = [var for _, var in fetch_vars.items()]
    print_interval = 50

    exe.train_from_dataset(
        program=paddle.static.default_main_program(),
        dataset=dataset,
        fetch_list=fetch_vars,
        fetch_info=fetch_info,
        print_period=print_interval,
        debug=False
    )


def dataset_test(epoch_id, dataset, fetch_vars, exe):
    fetch_info = [
        "Epoch {} Var {}".format(epoch_id, var_name) for var_name in fetch_vars
    ]
    fetch_vars = [var for _, var in fetch_vars.items()]

    exe.infer_from_dataset(
        program=paddle.static.default_main_program(),
        dataset=dataset,
        fetch_list=fetch_vars,
        fetch_info=fetch_info,
        print_period=1,
        debug=False
    )