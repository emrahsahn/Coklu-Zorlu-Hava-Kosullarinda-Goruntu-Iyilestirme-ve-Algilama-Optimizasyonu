"""
Device-agnostic loaders for HVI-CIDNet (low-light), FFA-Net (dehazing), and DDN (rain streaks).

FFA checkpoints may use ``DataParallel`` (``module.`` keys). HVI-CIDNet lives under ``HVI-CIDNet/``.
DDN (Deep Detailed Network) uses ``src/ddn_model.py`` with Rain1400 / Rain100L/H weights.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Absolute path to the project repository root."""
    return _REPO_ROOT


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _strip_dataparallel_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


# ---------------------------------------------------------------------------
# DDN — Deep Detailed Network (rain streak removal)
# ---------------------------------------------------------------------------

_DDN_WEIGHTS_DIR = _REPO_ROOT / "Deep_Detailed_Network-PyTorch-master" / "model"
DDNVariant = Literal["rain100L", "rain100H", "rain1400"]

_DDN_VARIANT_META: dict[DDNVariant, dict[str, str]] = {
    "rain1400": {
        "label": "Rain1400 fine-tune",
        "weights": "model/rain1400_finetune_new/model_best.pth",
        "task": "Rain1400 üzerinde fine-tune (12600 görüntü, genel yağmur sahneleri)",
        "notes": "Gerçek dünya yağmur görüntülerinde daha iyi genelleme; pipeline varsayılanı.",
    },
    "rain100L": {
        "label": "Rain100L",
        "weights": "model/rain100L/model_best.pth",
        "task": "Hafif yağmur çizgisi (Rain100L, PSNR ~33.6)",
        "notes": "Rain100L test setinde en yüksek PSNR; hafif yağmur için.",
    },
    "rain100H": {
        "label": "Rain100H",
        "weights": "model/rain100H/model_best.pth",
        "task": "Şiddetli yağmur çizgisi (Rain100H)",
        "notes": "Yoğun yağmur çizgileri için alternatif ağırlık.",
    },
}

DEFAULT_DDN_VARIANT: DDNVariant = "rain1400"


def _load_ddn_from_checkpoint(path: Path, device: torch.device) -> nn.Module:
    from . import ddn_model

    # Upstream checkpoints pickle DeRain as __main__.DeRain
    main = sys.modules.get("__main__")
    if main is not None and not hasattr(main, "DeRain"):
        setattr(main, "DeRain", ddn_model.DeRain)

    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, nn.DataParallel):
        return raw.module.to(device).eval()
    if hasattr(raw, "module"):
        return raw.module.to(device).eval()
    if isinstance(raw, nn.Module):
        return raw.to(device).eval()
    model = ddn_model.DeRain().to(device)
    state = _strip_dataparallel_prefix(raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state)
    model.eval()
    return model


class DDNWrapper:
    """
    Rain streak removal (DDN, CVPR 2017). RGB ``[B,3,H,W]`` in ``[0,1]`` → same shape, clamped.
    """

    _DEFAULT_WEIGHTS = {
        "rain1400": _DDN_WEIGHTS_DIR / "rain1400_finetune_new" / "model_best.pth",
        "rain100L": _DDN_WEIGHTS_DIR / "rain100L" / "model_best.pth",
        "rain100H": _DDN_WEIGHTS_DIR / "rain100H" / "model_best.pth",
    }

    def __init__(
        self,
        variant: DDNVariant = DEFAULT_DDN_VARIANT,
        weights_path: Optional[Path | str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.variant: DDNVariant = variant
        self.weights_path = (
            Path(weights_path) if weights_path else self._DEFAULT_WEIGHTS[variant]
        )
        self.device = device or default_device()
        self._model: Optional[nn.Module] = None

    def load(self) -> "DDNWrapper":
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"DDN weights not found: {self.weights_path}")
        self._model = _load_ddn_from_checkpoint(self.weights_path, self.device)
        return self

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Call load() before inference.")
        return self._model

    def restore(self, rgb_01: torch.Tensor) -> torch.Tensor:
        x = rgb_01.to(self.device, dtype=torch.float32).clamp(0, 1)
        with torch.no_grad():
            out = self.model(x)
        return torch.clamp(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# FFA-Net (dehazing)
# ---------------------------------------------------------------------------

_FFA_NET_DIR = _REPO_ROOT / "FFA-Net" / "FFA-Net" / "net"
_FFA_WEIGHTS_DIR = _REPO_ROOT / "FFA-Net" / "FFA-Net" / "trained_models"
FFATask = Literal["its", "ots"]


class FFANetWrapper:
    """
    FFA dehazing. ``task='its'`` uses indoor weights (blocks=20); ``task='ots'`` outdoor (blocks=19).
    Input: RGB ``[B,3,H,W]`` in ``[0,1]`` (ImageNet-style normalize inside). Output: RGB clamped.
    """

    _TASK_BLOCKS = {"its": 20, "ots": 19}

    def __init__(
        self,
        task: FFATask = "its",
        weights_path: Optional[Path | str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.task: FFATask = task
        self.device = device or default_device()
        blocks = self._TASK_BLOCKS[task]
        if weights_path is not None:
            self.weights_path = Path(weights_path)
        else:
            self.weights_path = _FFA_WEIGHTS_DIR / f"{task}_train_ffa_3_{blocks}.pk"
        self._blocks = blocks
        self._model: Optional[nn.Module] = None

    def load(self) -> "FFANetWrapper":
        if str(_FFA_NET_DIR) not in sys.path:
            sys.path.insert(0, str(_FFA_NET_DIR))

        if not self.weights_path.is_file():
            raise FileNotFoundError(f"FFA-Net weights not found: {self.weights_path}")

        from models.FFA import FFA  # type: ignore

        ckp = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        raw = ckp["model"] if isinstance(ckp, dict) and "model" in ckp else ckp
        state = _strip_dataparallel_prefix(raw) if isinstance(raw, dict) else raw

        net = FFA(gps=3, blocks=self._blocks).to(self.device)
        net.load_state_dict(state)
        net.eval()
        self._model = net
        return self

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Call load() before inference.")
        return self._model

    def restore(self, rgb_01: torch.Tensor) -> torch.Tensor:
        x = rgb_01.to(self.device, dtype=torch.float32).clamp(0, 1)
        mean = torch.tensor([0.64, 0.6, 0.58], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.14, 0.15, 0.152], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x_norm = (x - mean) / std
        with torch.no_grad():
            pred = self.model(x_norm)
        return torch.clamp(pred, 0.0, 1.0)


# ---------------------------------------------------------------------------
# HVI-CIDNet (low-light, upstream)
# ---------------------------------------------------------------------------

_HVI_REPO = _REPO_ROOT / "HVI-CIDNet"
_DEFAULT_HVI_WEIGHTS = _HVI_REPO / "weights" / "train" / "epoch_100.pth"


class HVI_CIDNetWrapper:
    """
    Original HVI-CIDNet from ``HVI-CIDNet/net/CIDNet.py``. Input RGB ``[B,3,H,W]`` in ``[0,1]``.
    Pads H,W to multiples of 8 (reflect), applies optional ``gamma`` on input (``input ** gamma``),
    crops back to original size. Sets ``gated`` / ``gated2`` and optional ``alpha_s`` / ``alpha``.
    """

    def __init__(
        self,
        weights_path: Optional[Path | str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else _DEFAULT_HVI_WEIGHTS
        self.device = device or default_device()
        self._model: Optional[nn.Module] = None

    def load(self) -> "HVI_CIDNetWrapper":
        root = str(_HVI_REPO)
        if root not in sys.path:
            sys.path.insert(0, root)

        if not self.weights_path.is_file():
            raise FileNotFoundError(f"HVI-CIDNet weights not found: {self.weights_path}")

        from net.CIDNet import CIDNet  # type: ignore

        model = CIDNet().to(self.device)
        state = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = _strip_dataparallel_prefix(state) if isinstance(state, dict) else state
        model.load_state_dict(state)
        model.eval()
        model.trans.gated = True  # type: ignore[attr-defined]
        model.trans.gated2 = True  # type: ignore[attr-defined]
        self._model = model
        return self

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Call load() before inference.")
        return self._model

    def restore(
        self,
        rgb_01: torch.Tensor,
        gamma: float = 1.0,
        alpha_s: float = 1.0,
        alpha_i: float = 1.0,
    ) -> torch.Tensor:
        m = self.model
        m.trans.alpha_s = alpha_s  # type: ignore[attr-defined]
        m.trans.alpha = alpha_i  # type: ignore[attr-defined]

        x = rgb_01.to(self.device, dtype=torch.float32).clamp(0, 1)
        b, _, h, w = x.shape
        factor = 8
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        x_in = torch.pow(x_pad, gamma)
        with torch.no_grad():
            out = m(x_in)
        out = torch.clamp(out, 0.0, 1.0)
        return out[:, :, :h, :w]


def ddn_variant_meta(variant: DDNVariant) -> dict[str, str]:
    """UI / rapor için DDN varyant açıklaması."""
    return dict(_DDN_VARIANT_META[variant])


__all__ = [
    "DDNWrapper",
    "DDNVariant",
    "DEFAULT_DDN_VARIANT",
    "FFANetWrapper",
    "HVI_CIDNetWrapper",
    "ddn_variant_meta",
    "default_device",
    "repo_root",
]
