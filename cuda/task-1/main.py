"""
Оптимизации:
- pin_memory + non_blocking перенос батчей на GPU — асинхронный конвейер подачи данных.
- Шум генерируется сразу на GPU (randn_like), без лишних CPU→GPU копий.
- zero_grad(set_to_none=True) — дешевле обнуление градиентов.
- Замер forward/backward через CUDA events + synchronize — корректные метрики времени.
- loss.item() вместо сохранения тензора — без утечки памяти из графа автограда.
"""

import statistics

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    device = torch.device('cuda')
    dataloader = DataLoader(
        prepare_data(),
        batch_size=256,
        shuffle=True,
        pin_memory=True
    )

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses_history = []
    forward_times = []
    backward_times = []

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        noise = torch.randn_like(data, device=device)
        data = data + noise

        optimizer.zero_grad(set_to_none=True)

        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        fwd_start.record()
        output = model(data)
        loss = criterion(output, target)
        fwd_end.record()

        bwd_start.record()
        loss.backward()
        bwd_end.record()
        optimizer.step()

        torch.cuda.synchronize()
        forward_times.append(fwd_start.elapsed_time(fwd_end) / 1000.0)
        backward_times.append(bwd_start.elapsed_time(bwd_end) / 1000.0)

        losses_history.append(loss.item())
        print(f"Batch {batch_idx} loss: {losses_history[-1]:.4f}")

    print(f"Epoch finished, avg forward time is {statistics.mean(forward_times)}, "
          f"avg backward time is {statistics.mean(backward_times)}")

if __name__ == '__main__':
    train()
