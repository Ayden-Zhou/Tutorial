import torch
import torch.nn.functional as F
import torch.distributed as dist
from parallel_utils import setup, cleanup, get_device, int_divide, get_init_params, summarize_tensor

def pipeline_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)

    # 数据准备
    data = data.to(get_device(rank))
    batch_size = data.size(0)
    num_dim = data.size(1)

    # --- 1. 层分片 ---
    # 将总层数平分给各个 rank
    local_num_layers = int_divide(num_layers, world_size)
    
    # 每个 rank 初始化属于自己的那部分层
    # Rank 0: Layers 0~1; Rank 1: Layers 2~3 (假设 total=4, world=2)
    local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]

    # --- 2. Micro-batch 切分 ---
    # 为了减少流水线气泡，将大 Batch 切小
    micro_batch_size = int_divide(batch_size, num_micro_batches)
    
    if rank == 0:
        # 只有 Rank 0 拥有原始输入数据
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # 后续 Rank 只需要分配接收缓存
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank)) 
                         for _ in range(num_micro_batches)]

    # --- 3. 流水线执行 ---
    for i, x in enumerate(micro_batches):
        # A. 接收 (Recv)
        # 如果不是第一个 stage，需要从上一个 rank 接收数据
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # B. 计算 (Compute)
        # 执行本 rank 负责的层
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # C. 发送 (Send)
        # 如果不是最后一个 stage，将结果发给下一个 rank
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: micro-batch {i} "
                  f"sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)
            dist.send(tensor=x, dst=rank + 1)
        else:
            # 最后一个 rank 输出最终结果
            if i == 0: # 只打印一次作为示例
                 print(f"[pipeline_parallelism] Rank {rank}: micro-batch {i} "
                       f"final output {summarize_tensor(x)}", flush=True)

    print("注：此处未展示通信与计算的重叠（Overlap），那是消除气泡的关键优化。")

    cleanup()
