import torch
import torch.nn.functional as F
import torch.distributed as dist
from parallel_utils import setup, cleanup, get_device, int_divide, get_init_params, summarize_tensor

def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)

    # --- 1. 数据分片 ---
    # 每个 rank 获取数据的一个切片
    # 在实际工程中，这里通常通过 DistributedSampler 来处理
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = int_divide(batch_size, world_size)
    
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size
    
    # 将属于本 rank 的数据移动到 GPU
    local_data = data[start_index:end_index].to(get_device(rank))

    # --- 2. 模型初始化 ---
    # 关键点：每个 rank 都持有完整的参数副本 (params)
    # 并且初始化必须完全一致（这里通过固定 seed 实现）
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    for step in range(num_steps):
        # --- 3. 前向传播 (Local) ---
        x = local_data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        
        # Loss 是基于本地数据计算的，因此每个 rank 的 Loss 值不同
        loss = x.square().mean()

        # --- 4. 反向传播 (Local) ---
        loss.backward()

        # --- 5. 梯度同步 (Global) ---
        # 这就是数据并行的核心：平均所有 worker 的梯度
        # 这样能等效于在整个大 Batch 上进行的训练
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # --- 6. 参数更新 ---
        # 由于梯度已经同步，所有 rank 的 optimizer step 行为将完全一致
        optimizer.step()

        print(f"[data_parallelism] Rank {rank}: step = {step}, "
              f"loss = {loss.item():.4f}, "
              f"params_sample = {[summarize_tensor(params[i]) for i in range(num_layers)]}", 
              flush=True)

    cleanup()
