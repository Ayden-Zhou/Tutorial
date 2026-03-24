import torch.distributed as dist
from inspect import isfunction
from typing import Callable
import sys
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import math
def get_device(rank):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")

class DisableDistributed:
    """Context manager that temporarily disables distributed functions (replaces with no-ops)"""
    def __enter__(self):
        self.old_functions = {}
        for name in dir(dist):
            value = getattr(dist, name, None)
            if isfunction(value):
                self.old_functions[name] = value
                setattr(dist, name, lambda *args, **kwargs: None)

    def __exit__(self, exc_type, exc_value, traceback):
        for name in self.old_functions:
            setattr(dist, name, self.old_functions[name])


def spawn(func: Callable, world_size: int, *args, **kwargs):
    # Note: assume kwargs are in the same order as what main needs
    if sys.gettrace():
        # If we're being traced, run the function directly, since we can't trace through mp.spawn
        with DisableDistributed():
            args = (0, world_size,) + args + tuple(kwargs.values())
            func(*args)
    else:
        args = (world_size,) + args + tuple(kwargs.values())
        mp.spawn(func, args=args, nprocs=world_size, join=True)


def int_divide(a: int, b: int):
    """Return a / b and throw an error if there's a remainder."""
    assert a % b == 0
    return a // b

def summarize_tensor(tensor: torch.Tensor) -> str:
    return "x".join(map(str, tensor.shape)) + "[" + str(round(tensor.view(-1)[0].item(), 4)) + "...]"


def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> nn.Parameter:
    torch.random.manual_seed(0)  # For reproducibility
    return nn.Parameter(torch.randn(num_inputs, num_outputs, device=get_device(rank)) / math.sqrt(num_outputs))


import os

def setup(rank: int, world_size: int):
    """
    初始化分布式进程组。
    指定 Master 节点地址和端口，用于协调（实际数据传输走 NCCL）。
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        # GPU 环境通常使用 NCCL 后端
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank) # 关键：设置当前进程使用的 GPU 设备
    else:
        # CPU 环境使用 Gloo 后端
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def render_duration(duration: float) -> str:
    if duration < 1e-3:
        return f"{duration * 1e6:.2f}us"
    if duration < 1:
        return f"{duration * 1e3:.2f}ms"
    return f"{duration:.2f}s"
