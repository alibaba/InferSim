from config.model_config import ModelConfig
from hardware.gpu import GPU


class Comm:
    def __init__(
        self,
        config: ModelConfig,
        gpu: GPU,
        world_size: int,
        num_nodes=1,
        enable_deepep=False,
    ):
        self.config = config
        self.gpu = gpu
        self.world_size = world_size
        self.num_nodes = num_nodes
        self.enable_deepep = enable_deepep

    def size_bw_model(self, tensor_shape, use_fp8=False, inter_node=False):
        if self.world_size <= 1:
            return 0
        size = 1 if use_fp8 else 2
        for v in tensor_shape:
            size *= v
        if inter_node:
            return size / (1024**3) / self.gpu.rdma_bw
        return size / (1024**3) / self.gpu.nvlink_bw

    def all_reduce(self, num_tokens):
        tensor_shape = [num_tokens * self.world_size, self.config.hidden_size]
        return self.size_bw_model(
            tensor_shape, use_fp8=False, inter_node=(self.num_nodes > 1)
        )

    def dispatch(self, num_tokens, mode="normal"):
        if mode == "normal":
            send_tokens = num_tokens * (self.num_nodes - 1)
            tensor_shape1 = [send_tokens, self.config.hidden_size]
            t1 = self.size_bw_model(tensor_shape1, use_fp8=True, inter_node=True)

            tensor_shape2 = [num_tokens, self.config.hidden_size]
            t2 = self.size_bw_model(tensor_shape2, use_fp8=True, inter_node=False)
            return t1 + t2
        else:
            send_tokens = num_tokens * self.config.num_experts_per_tok
            tensor_shape = [send_tokens, self.config.hidden_size]
            return self.size_bw_model(
                tensor_shape, use_fp8=True, inter_node=(self.num_nodes > 1)
            )

    def combine(self, num_tokens, mode="normal"):
        if mode == "normal":
            rcv_tokens = num_tokens * (self.num_nodes - 1)
            tensor_shape1 = [rcv_tokens, self.config.hidden_size]
            t1 = self.size_bw_model(tensor_shape1, use_fp8=False, inter_node=True)

            tensor_shape2 = [num_tokens, self.config.hidden_size]
            t2 = self.size_bw_model(tensor_shape2, use_fp8=False, inter_node=False)
            return t1 + t2
        else:
            rcv_tokens = num_tokens * self.config.num_experts_per_tok
            tensor_shape = [rcv_tokens, self.config.hidden_size]
            return self.size_bw_model(
                tensor_shape, use_fp8=False, inter_node=(self.num_nodes > 1)
            )

    def a2f(self, num_tokens):
        tensor_shape = [num_tokens, self.config.hidden_size]
        return self.size_bw_model(tensor_shape, use_fp8=True, inter_node=True)

    def f2a(self, num_tokens):
        tensor_shape = [num_tokens, self.config.hidden_size]
        return self.size_bw_model(tensor_shape, use_fp8=False, inter_node=True)

    def prefill_comm(self, num_tokens: int):
        if self.enable_deepep:
            return self.dispatch(num_tokens, "normal"), self.combine(
                num_tokens, "normal"
            )
        return self.all_reduce(num_tokens), self.all_reduce(num_tokens)

    def decode_comm(self, num_tokens: int):
        if self.enable_deepep:
            return self.dispatch(num_tokens, "low_latency"), self.combine(
                num_tokens, "low_latency"
            )
        return self.all_reduce(num_tokens), self.all_reduce(num_tokens)

    def tp_all_reduce(self, num_tokens: int, tp_size: int):
        """Calculate TP all_reduce communication time.

        TP all_reduce happens after attention and MoE computation within each TP group.
        For TP=N, each GPU holds 1/N of the output and needs to all_reduce to get full result.
        """
        if tp_size <= 1:
            return 0
        # TP all_reduce: each GPU sends/receives (tp_size - 1) / tp_size of data
        # Using ring all_reduce: 2 * (tp_size - 1) / tp_size of data transferred
        tensor_shape = [num_tokens, self.config.hidden_size]
        size = 2  # fp16/bf16 output
        for v in tensor_shape:
            size *= v
        # Ring all_reduce: 2 * (N-1)/N data transfer
        transfer_size = size * 2 * (tp_size - 1) / tp_size
        # Use NVLink bandwidth for intra-node TP communication
        return transfer_size / (1024**3) / self.gpu.nvlink_bw
