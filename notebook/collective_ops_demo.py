import torch
import torch.distributed as dist
from parallel_utils import setup, cleanup, get_device

def collective_operations_main(rank: int, world_size: int):
    """
    此函数将在每个进程（rank = 0, ..., world_size - 1）中异步运行。
    """
    setup(rank, world_size)

    # --- 1. All-reduce 演示 ---
    dist.barrier()  # 等待所有进程到达此处，为了打印整齐

    # 创建 tensor: Rank 0 -> [0, 1, 2, 3], Rank 1 -> [1, 2, 3, 4] ...
    tensor = torch.tensor([0., 1, 2, 3], device=get_device(rank)) + rank 
    
    print(f"Rank {rank} [before all-reduce]: {tensor}", flush=True)
    
    # 执行 All-reduce (Sum)，原地修改 tensor
    # 结果应该是所有 rank 对应位置的元素之和
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    
    print(f"Rank {rank} [after all-reduce]: {tensor}", flush=True)

    # --- 2. Reduce-scatter 演示 ---
    dist.barrier()

    # 输入数据: 每个 rank 持有 world_size 长度的数据
    input = torch.arange(world_size, dtype=torch.float32, device=get_device(rank)) + rank
    # 输出数据: 每个 rank 只接收归约后的一部分（标量）
    output = torch.empty(1, device=get_device(rank)) 

    print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
    
    # 结果：Rank i 将持有所有 input[i] 的和
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    
    print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)

    # --- 3. All-gather 演示 ---
    dist.barrier()

    # 将上一步 Reduce-scatter 的输出作为输入
    input_tensor = output
    # 输出: 收集所有 rank 的数据，恢复成向量
    output_tensor = torch.empty(world_size, device=get_device(rank)) 

    print(f"Rank {rank} [before all-gather]: input = {input_tensor}, output = {output_tensor}", flush=True)
    
    dist.all_gather_into_tensor(output_tensor=output_tensor, input_tensor=input_tensor, async_op=False)
    
    print(f"Rank {rank} [after all-gather]: input = {input_tensor}, output = {output_tensor}", flush=True)

    # 验证结论：All-reduce = Reduce-scatter + All-gather
    
    cleanup()
