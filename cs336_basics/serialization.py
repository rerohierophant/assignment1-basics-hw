import torch
import os
from typing import IO, BinaryIO

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    给定模型、优化器和迭代次数，将它们序列化到磁盘。

    Args:
        model (torch.nn.Module): 序列化此模型的状态。
        optimizer (torch.optim.Optimizer): 序列化此优化器的状态。
        iteration (int): 序列化此值，它表示我们已经完成的训练迭代次数。
        out (str | os.PathLike | BinaryIO | IO[bytes]): 用于序列化模型、优化器和迭代次数的路径或类文件对象。
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    给定序列化的检查点（路径或类文件对象），将序列化的状态恢复到给定的模型和优化器。
    返回我们之前在检查点中序列化的迭代次数。

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): 序列化检查点的路径或类文件对象。
        model (torch.nn.Module): 恢复此模型的状态。
        optimizer (torch.optim.Optimizer): 恢复此优化器的状态。
    Returns:
        int: 之前序列化的迭代次数。
    """
    # 加载检查点
    checkpoint = torch.load(src, map_location="cpu") # map_location="cpu" 确保即使在 GPU 上保存也能加载

    # 恢复模型和优化器状态
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]
