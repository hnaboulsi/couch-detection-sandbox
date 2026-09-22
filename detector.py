"""The single Grounding DINO detector used by the minimal experiment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    bbox: tuple[float, float, float, float]
    model_name: str


class Detector(Protocol):
    def detect(self, image: np.ndarray, prompts: Sequence[str]) -> list[Detection]: ...

    def metadata(self) -> dict[str, Any]: ...


class GroundingDINO:
    model_name = "IDEA-Research/grounding-dino-tiny"
    revision = "a2bb814dd30d776dcf7e30523b00659f4f141c71"

    def __init__(self, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Install the dependencies listed in requirements.txt"
            ) from error
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self._torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, revision=self.revision
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_name, revision=self.revision
        ).to(device)
        self.model.eval()

    def detect(self, image: np.ndarray, prompts: Sequence[str]) -> list[Detection]:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                "image must be an uint8 RGB array with shape (height, width, 3)"
            )
        prompt_list = [prompt.strip().lower() for prompt in prompts]
        if not prompt_list or any(not prompt for prompt in prompt_list):
            raise ValueError("prompts must contain non-empty labels")
        inputs = self.processor(
            images=image,
            text=[prompt_list],
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.35,
            text_threshold=0.35,
            target_sizes=[image.shape[:2]],
            text_labels=[prompt_list],
        )[0]
        return [
            Detection(
                label=str(label),
                score=float(score.item()),
                bbox=tuple(float(value) for value in box.tolist()),
                model_name=self.model_name,
            )
            for box, score, label in zip(
                result["boxes"], result["scores"], result["text_labels"], strict=True
            )
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "device": self.device,
            "box_threshold": 0.35,
            "text_threshold": 0.35,
        }
