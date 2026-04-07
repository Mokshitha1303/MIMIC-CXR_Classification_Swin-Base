"""
Model builder for MIMIC-CXR classification.

Primary backbone: Swin Transformer Base from timm
  - Default init: ImageNet-21k pretrained weights via timm
  - Optional:     ARK+ pretrained CXR weights (load checkpoint key 'teacher')

ARK+ weight loading protocol (from ARK+ README):
  1. Load checkpoint from path
  2. Extract state_dict from checkpoint['teacher']
  3. Strip 'module.' prefixes
  4. Delete 'head.*' and 'attn_mask' keys
  5. load_state_dict(strict=False)
"""

import timm
import torch
import torch.nn as nn


def build_model(cfg) -> nn.Module:
    """Build Swin_Base classification model.

    Args:
        cfg: config object with attributes:
            cfg.model.name              — timm model identifier
            cfg.model.pretrained        — bool, load ImageNet weights from timm
            cfg.model.pretrained_weights — optional path to ARK+ checkpoint
            cfg.model.ark_checkpoint_key — key in ARK+ checkpoint (default 'teacher')
            cfg.data.num_classes        — number of output classes
    Returns:
        nn.Module ready for training / inference
    """
    num_classes = cfg.data.num_classes
    model_name  = cfg.model.name

    if cfg.model.pretrained_weights:
        # ------------------------------------------------------------------ #
        # Load ARK+ (or custom) pretrained weights
        # ------------------------------------------------------------------ #
        print(f"[build_model] Creating {model_name} (no timm pretrain) …")
        model = timm.create_model(model_name, num_classes=num_classes, pretrained=False)
        _load_pretrained_weights(
            model,
            cfg.model.pretrained_weights,
            checkpoint_key=cfg.model.ark_checkpoint_key,
        )
    else:
        # ------------------------------------------------------------------ #
        # Standard ImageNet-21k pretrained via timm
        # ------------------------------------------------------------------ #
        use_pretrained = cfg.model.pretrained
        print(
            f"[build_model] Creating {model_name} "
            f"(pretrained={use_pretrained}) …"
        )
        model = timm.create_model(
            model_name,
            num_classes=num_classes,
            pretrained=use_pretrained,
        )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[build_model] Parameters: {n_params:.1f}M total, {n_train:.1f}M trainable")
    return model


def _load_pretrained_weights(
    model: nn.Module,
    pretrained_weights: str,
    checkpoint_key: str = 'teacher',
) -> None:
    """Load pretrained weights into model (ARK+ protocol)."""
    print(f"[build_model] Loading pretrained weights from {pretrained_weights} …")

    checkpoint = torch.load(pretrained_weights, map_location='cpu')
    print(f"[build_model] Checkpoint keys: {list(checkpoint.keys())}")

    # Extract state dict
    if checkpoint_key and checkpoint_key in checkpoint:
        state_dict = checkpoint[checkpoint_key]
        print(f"[build_model] Using checkpoint key '{checkpoint_key}'")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Strip 'module.' prefix (DataParallel / DistributedDataParallel artifact)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Remove keys that should not be loaded (classification head, positional masks)
    keys_to_delete = [
        k for k in state_dict
        if k.startswith('head.') or 'attn_mask' in k
    ]
    if keys_to_delete:
        print(f"[build_model] Removing keys: {keys_to_delete}")
        for k in keys_to_delete:
            del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[build_model] Loaded with msg: {msg}")
