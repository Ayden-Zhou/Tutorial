import time
import torch
import torch.distributed as dist
from parallel_utils import setup, cleanup, get_device, render_duration

def reduce_scatter_benchmark(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # 创建输入：每个 Rank 都有一个 (world_size, num_elements) 的矩阵
    # 也就是说，输入数据量是 all-reduce 的 world_size 倍
    input = torch.randn(world_size, num_elements, device=get_device(rank)) 
    output = torch.empty(num_elements, device=get_device(rank))

    # Warmup
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        dist.barrier()

    # 计时
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        dist.barrier()
    end_time = time.time()

    duration = end_time - start_time
    # 只打印 Rank 0 的时间
    if rank == 0:
        print(f"[reduce_scatter] Rank {rank}: duration {render_duration(duration)}", flush=True)

    # 计算带宽
    dist.barrier()
    data_bytes = input.element_size() * input.numel()
    # Reduce-scatter 只需要发送数据进行归约，不需要像 All-reduce 那样再广播结果
    # 发送量系数没有 2x
    sent_bytes = data_bytes * (world_size - 1)
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    
    if rank == 0:
        print(f"[reduce_scatter] Measured Bandwidth = {round(bandwidth / 1024**3, 2)} GB/s", flush=True)

    cleanup()
