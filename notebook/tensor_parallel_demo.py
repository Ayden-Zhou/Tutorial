import torch
import torch.nn.functional as F
import torch.distributed as dist
from parallel_utils import setup, cleanup, get_device, int_divide, get_init_params, summarize_tensor

def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)

    # 假设输入数据对所有 rank 可见（或者这里我们为了演示让所有 rank 拥有全量数据）
    data = data.to(get_device(rank))
    batch_size = data.size(0)
    num_dim = data.size(1)
    
    # 确定分片大小：将特征维度 num_dim 切分
    local_num_dim = int_divide(num_dim, world_size)

    # --- 1. 模型分片 ---
    # 每个 rank 只持有 1/world_size 的参数
    # 权重形状为: [num_dim, local_num_dim]
    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]

    # --- 2. 前向传播 ---
    x = data
    for i in range(num_layers):
        # Local Compute: (batch, num_dim) @ (num_dim, local_num_dim) -> (batch, local_num_dim)
        # 这一步只计算了输出向量的一部分切片
        x = x @ params[i] 
        x = F.gelu(x)

        # Communication: 需要收集所有 rank 的计算结果来恢复完整的向量
        # 准备缓冲区
        activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank)) 
                       for _ in range(world_size)]

        # All-gather: 每个 rank 将自己的结果广播给所有人
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # 拼接: 将 [slice_1, slice_2, ...] 拼回 [full_vector]
        # 结果形状回到: (batch_size, num_dim)
        x = torch.cat(activations, dim=1)

    print(f"[tensor_parallelism] Rank {rank}: output shape {summarize_tensor(x)}", flush=True)

    # 注意：反向传播需要推导对应的梯度切分逻辑（此处留作练习）

    cleanup()
