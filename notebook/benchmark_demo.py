import time
import torch
import torch.distributed as dist
from parallel_utils import setup, cleanup, get_device, render_duration

def all_reduce_benchmark(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # 创建一个大 Tensor
    tensor = torch.randn(num_elements, device=get_device(rank))

    # 1. Warmup (预热)
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()

    # 2. 计时
    start_time = time.time()
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # 注意：计时结束前不需要再 barrier，因为 all_reduce 本身包含隐式同步（对于结果正确性而言），
    # 但为了测量所有进程完成的时间，barrier 是合理的。
    dist.barrier() 
    end_time = time.time()

    duration = end_time - start_time
    # 只打印 Rank 0 的时间，避免刷屏
    if rank == 0:
        print(f"[all_reduce] Rank {rank}: duration {render_duration(duration)}", flush=True)

    # 3. 计算有效带宽 (Bus Bandwidth)
    # Ring All-reduce 的总线带宽公式： B = (2 * (N-1) / N) * Size / Time
    size_bytes = tensor.element_size() * tensor.numel()
    # 修正后的带宽计算公式
    bus_bandwidth = (2 * (world_size - 1) / world_size) * size_bytes / duration
    
    if rank == 0:
        print(f"[all_reduce] Size: {size_bytes/1024**2:.2f} MB")
        print(f"[all_reduce] Bus Bandwidth = {bus_bandwidth / 1024**3:.2f} GB/s", flush=True)

    cleanup()
