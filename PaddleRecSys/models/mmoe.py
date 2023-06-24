import paddle
import paddle.nn.functional as F


class MMoE(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_dim = self.config["models"]["mmoe"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["mmoe"]["sparse_inputs_slots"] - 2
        self.expert_num = self.config["models"]["mmoe"]["expert_num"]  # number of expert networks
        self.expert_size = self.config["models"]["mmoe"]["expert_size"]  # expert network output size
        self.tower_size = self.config["models"]["mmoe"]["tower_size"]  # tower network output size
        self.gate_num = self.config["models"]["mmoe"]["gate_num"]  # number of gate networks

        self.experts = []
        expert_init = [pow(10, -i) for i in range(1, self.expert_num + 1)]

        for i in range(self.expert_num):
            linear = paddle.nn.Linear(
                in_features=self.sparse_feature_dim * self.num_field,
                out_features=self.expert_size,
                weight_attr=paddle.nn.initializer.Constant(value=expert_init[i]),
                bias_attr=paddle.nn.initializer.Constant(value=0.1)
            )

            self.experts.append(linear)

        self.gates = []
        self.towers = []
        self.tower_outs = []

        gate_init = [pow(10, -i) for i in range(1, self.gate_num + 1)]

        for i in range(self.gate_num):
            curr_gate = paddle.nn.Linear(
                in_features=self.sparse_feature_dim * self.num_field,
                out_features=self.expert_num,
                weight_attr=paddle.nn.initializer.Constant(value=gate_init[i]),
                bias_attr=paddle.nn.initializer.Constant(value=0.1)
            )

            self.gates.append(curr_gate)

            curr_tower = paddle.nn.Linear(
                in_features=self.expert_size,
                out_features=self.tower_size,
                weight_attr=paddle.nn.initializer.Constant(value=gate_init[i]),
                bias_attr=paddle.nn.initializer.Constant(value=0.1)
            )

            self.towers.append(curr_tower)

            curr_tower_out = paddle.nn.Linear(
                in_features=self.tower_size,
                out_features=2,
                weight_attr=paddle.nn.initializer.Constant(value=gate_init[i]),
                bias_attr=paddle.nn.initializer.Constant(value=0.1)
            )

            self.tower_outs.append(curr_tower_out)

    def forward(self, sparse_embs):
        x = paddle.concat(sparse_embs, axis=1)

        expert_outputs = []
        for i in range(self.expert_num):
            linear_out = self.experts[i](x)
            expert_output = F.relu(linear_out)
            expert_outputs.append(expert_output)

        expert_concat = paddle.concat(expert_outputs, axis=1)
        expert_concat = paddle.reshape(expert_concat, shape=[-1, self.expert_num, self.expert_size])

        output_layers = []
        for i in range(self.gate_num):
            curr_gate_linear = self.gates[i](x)
            curr_gate = F.softmax(curr_gate_linear)
            curr_gate = paddle.reshape(curr_gate, [-1, self.expert_num, 1])

            curr_gate_expert = paddle.multiply(x=expert_concat, y=curr_gate)
            curr_gate_expert = paddle.sum(curr_gate_expert, axis=1)

            curr_tower = self.towers[i](curr_gate_expert)
            curr_tower = F.relu(curr_tower)

            out = self.tower_outs[i](curr_tower)
            out = F.softmax(out)
            out = paddle.clip(out, min=1e-15, max=1.0 - 1e-15)
            output_layers.append(out)

        return output_layers
