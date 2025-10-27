#!/usr/bin/env python3
import torch
import time

def test_bandwidth(device="cuda", size_gb=2, num_iters=10):
    """
    测试显存带宽 (GB/s)
    :param device: 测试的设备 ("cuda" 或 "cpu")
    :param size_gb: 测试数组大小（以 GB 计）
    :param num_iters: 测试迭代次数
    """
    assert torch.cuda.is_available(), "CUDA 不可用！"
    torch.cuda.synchronize()
    dtype = torch.float32

    size = int(size_gb * (1024**3) / 4)  # float32 -> 4 bytes
    print(f"Testing {size_gb} GB buffer on {device}...")

    # 分配显存
    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    torch.cuda.synchronize()

    # 预热
    for _ in range(3):
        c = a + b
    torch.cuda.synchronize()

    # 正式测试
    start = time.time()
    for _ in range(num_iters):
        c = a + b  # 读 a,b 写 c
    torch.cuda.synchronize()
    end = time.time()

    elapsed = (end - start) / num_iters
    bytes_moved = size * 4 * 3  # a,b,c 三个 tensor 都访问
    bandwidth = bytes_moved / elapsed / (1024**3)
    print(f"Average elapsed time: {elapsed*1000:.2f} ms")
    print(f"Estimated bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    test_bandwidth("cuda", size_gb=2, num_iters=10)
