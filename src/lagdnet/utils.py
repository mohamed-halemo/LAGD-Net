"""Temperature schedule, gate state, evaluation, latency and MACs."""

import time

import torch
from thop import profile


def get_current_temp(epoch, total_epochs, start_temp, end_temp):
    """T_e = start * (end / start) ** (e / total): 5 * 0.02 ** (e / 50) in the paper."""
    decay = (end_temp / start_temp) ** (1 / total_epochs)
    return start_temp * (decay ** epoch)


def gated_blocks(model):
    """The gated residual blocks of a network, in order."""
    return [m for m in model.modules() if hasattr(m, 'gate') and hasattr(m, 'snapped')]


def gate_state_dict(model):
    """Gate temperatures and snapped flags, which state_dict() does not hold."""
    return [{'logit': float(b.gate.logit.detach().cpu().item()),
             'temperature': float(b.gate.temperature),
             'snapped': bool(b.snapped)} for b in gated_blocks(model)]


def load_gate_state(model, gate_state):
    blocks = gated_blocks(model)
    if len(gate_state) != len(blocks):
        raise ValueError(f'gate state has {len(gate_state)} entries, the network has {len(blocks)} blocks')
    for b, s in zip(blocks, gate_state):
        b.gate.temperature = s['temperature']
        b.snapped = s['snapped']


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Mean loss and accuracy (%) over a full evaluation split."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss_sum += criterion(outputs, targets).item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return loss_sum / total, 100.0 * correct / total


def latency_and_macs(model, device, input_shape):
    """Latency (ms) of one image over 500 passes after 20 warm-up passes, with CUDA
    synchronization, and the multiply-accumulates (GMACs) counted by thop."""
    model.eval()
    x = torch.randn(1, *input_shape).to(device)
    macs, _ = profile(model, inputs=(x,), verbose=False)
    for _ in range(20):
        model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(500):
        model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - start) / 500 * 1000, macs / 1e9
