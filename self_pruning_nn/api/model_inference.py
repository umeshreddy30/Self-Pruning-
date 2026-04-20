from __future__ import annotations

import os
from io import BytesIO
from typing import Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

from models import SelfPruningNet
from utils import CIFAR10_MEAN, CIFAR10_STD


def _default_checkpoint_path() -> str:
    # matches train.py naming: outputs/best_model_lam{lam_tag}.pt
    return os.getenv("SELF_PRUNING_CKPT", os.path.join(".", "outputs", "best_model_lam0_0.pt"))


def load_model(device: torch.device, checkpoint_path: Optional[str] = None) -> SelfPruningNet:
    ckpt = checkpoint_path or _default_checkpoint_path()
    model = SelfPruningNet().to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


_preprocess = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
)


@torch.inference_mode()
def predict_image_bytes(
    model: SelfPruningNet, device: torch.device, image_bytes: bytes
) -> Tuple[int, float]:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = _preprocess(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    conf, pred = torch.max(probs, dim=0)
    return int(pred.item()), float(conf.item())

