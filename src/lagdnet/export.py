"""Compact export: every removed block is replaced by its shortcut module (the identity, or the
strided 1x1 projection at a stage entry). The export computes the same function as the network
after the threshold, and its state_dict holds only the remaining blocks."""

from lagdnet.layers import TAU
from lagdnet.utils import gated_blocks


def closed_indices(model):
    """Indices of the blocks removed by the threshold."""
    return [i for i, b in enumerate(gated_blocks(model)) if b.snapped and b.gate.value() < TAU]


def apply_removal(model, remove_idx):
    """Replace the listed blocks with their shortcut modules."""
    blocks = gated_blocks(model)
    targets = [blocks[i] for i in sorted(set(remove_idx))]
    replaced = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if any(child is t for t in targets):
                setattr(parent, name, child.shortcut)
                replaced += 1
    if replaced != len(targets):
        raise RuntimeError(f'replaced {replaced} blocks, expected {len(targets)}')
    return model
