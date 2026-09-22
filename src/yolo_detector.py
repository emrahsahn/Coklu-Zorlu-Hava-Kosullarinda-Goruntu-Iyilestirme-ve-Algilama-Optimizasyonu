"""
YOLOv8 (Ultralytics) inference: bounding boxes, labels, and JSON-serializable detections.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PIL import Image

from .model_loader import default_device, repo_root


class YoloV8Detector:
    """
    Loads ``yolov8n.pt`` from the repo root by default. ``predict`` accepts RGB ``uint8`` HWC numpy
    or a ``PIL.Image``; returns annotated RGB image and a list of detection dicts.
    """

    def __init__(
        self,
        weights_path: Optional[Path | str] = None,
        device: Optional[Any] = None,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else repo_root() / "yolov8n.pt"
        self.device = device or default_device()
        self._model = None

    def load(self) -> "YoloV8Detector":
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights_path}")

        from ultralytics import YOLO  # type: ignore

        self._model = YOLO(str(self.weights_path))
        return self

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Call load() before predict().")
        return self._model

    def _predict_device(self) -> str | int:
        if self.device.type == "cuda":
            return 0
        return "cpu"

    def predict(
        self,
        image: np.ndarray | Image.Image,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> dict[str, Any]:
        """
        :param image: RGB ``uint8`` ``HWC`` numpy array, or ``PIL.Image`` (RGB).
        :return: ``annotated_rgb`` (``uint8`` ``HWC`` RGB), ``detections`` (list of dicts),
                 ``num_detections``, ``json`` (pretty-printed string).
        """
        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        else:
            rgb = np.asarray(image)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("image must be HWC RGB with 3 channels")
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)

        results = self.model.predict(
            source=rgb,
            conf=float(conf),
            iou=float(iou),
            verbose=False,
            device=self._predict_device(),
        )
        r = results[0]
        plotted_bgr = r.plot()
        annotated_rgb = np.ascontiguousarray(plotted_bgr[:, :, ::-1])

        detections: List[dict[str, Any]] = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names
            for i in range(xyxy.shape[0]):
                cid = int(cls_ids[i])
                detections.append(
                    {
                        "class_id": cid,
                        "class_name": str(names.get(cid, str(cid))),
                        "confidence": float(confs[i]),
                        "xyxy": [float(x) for x in xyxy[i].tolist()],
                    }
                )

        payload = {
            "annotated_rgb": annotated_rgb,
            "detections": detections,
            "num_detections": len(detections),
        }
        payload["json"] = json.dumps(detections, indent=2, ensure_ascii=False)
        return payload


def detections_to_json(detections: List[dict[str, Any]], *, indent: int = 2) -> str:
    return json.dumps(detections, indent=indent, ensure_ascii=False)


__all__ = ["YoloV8Detector", "detections_to_json"]
