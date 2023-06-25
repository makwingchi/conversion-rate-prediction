import paddle


class Queue:
    def __init__(self, input_var, file_list, config):
        assert isinstance(input_var, list)
        assert len(file_list) > 0

        self.config = config

        self.input_var = input_var
        self.file_list = file_list

        self.pipe_command = self.config["runner"]["pipe_command"]
        assert self.pipe_command != None

        print("pipe_command is: {}".format(self.pipe_command))

        self.batch_size = self.config["runner"]["batch_size"]
        assert self.batch_size >= 1

        self.thread_num = self.config["runner"]["thread_num"]
        print("dataset init thread_num:", self.thread_num)
        assert self.thread_num >= 1

        self.infer_batch_size = self.config["runner"]["batch_size"]
        self.infer_thread_num = self.thread_num

    def get_reader(self):
        print("Get Train Dataset")
        dataset = paddle.distributed.QueueDataset()
        dataset.init(
            use_var=self.input_var,
            pipe_command=self.pipe_command,
            batch_size=self.batch_size,
            thread_num=self.thread_num
        )
        print("dataset get_reader thread_num:", self.thread_num)
        dataset.set_filelist(self.file_list)
        return dataset

    def get_infer_reader(self):
        print("Get Infer Dataset")
        dataset = paddle.distributed.QueueDataset()

        dataset.init(
            use_var=self.input_var,
            pipe_command=self.pipe_command,
            batch_size=self.infer_batch_size,
            thread_num=self.infer_thread_num
        )
        print("dataset get_infer_reader thread_num:", self.infer_thread_num)
        dataset.set_filelist(self.file_list)
        return dataset
