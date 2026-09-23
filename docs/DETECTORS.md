# Detector adapters and evaluation

The detector adapter is the boundary between a model's API and this experiment. MuJoCo produces an RGB image and a separate ground-truth mask. The detector receives **only** the RGB array and text prompts; it cannot inspect the mask. `run.py` evaluates the adapter's returned boxes afterward. Grounding DINO is the only model implemented in this repository, but its output uses a small shared record so the rest of the experiment can work with another detector later.

## The contract

`detector.py` defines this shape:

```python
@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    bbox: tuple[float, float, float, float]
    model_name: str


class Detector(Protocol):
    def detect(self, image: np.ndarray, prompts: Sequence[str]) -> list[Detection]: ...
    def metadata(self) -> dict[str, Any]: ...
```

`image` is an `H × W × 3` NumPy array of `uint8` RGB pixels. `prompts` is a sequence of labels, currently `[SCENE.prompt]`. Return one `Detection` per predicted box. `bbox` is `(left, top, right, bottom)` in **pixel coordinates** of that same image, with the origin at its upper-left corner; it is not normalized and is not centre-width-height format. `score` should be a finite confidence value in `[0, 1]`, with larger meaning more confident. `label` must identify the requested object in the prompt; the current evaluator compares it to ground truth case-insensitively but does not resolve synonyms. `model_name` identifies the producer of each box. `metadata()` should return JSON-serializable information sufficient to identify the model, revision/weights, inference thresholds, and device used for the run.

The protocol is structural: a class with those two methods can be passed to `run_experiment` without inheriting from `Detector`. `run.py` constructs `GroundingDINO` explicitly in `main`; there is no registry or auto-discovery. This is intentional for one working model.

## What the Grounding DINO adapter does

`GroundingDINO.__init__` pins `IDEA-Research/grounding-dino-tiny` to revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. It loads Hugging Face's `AutoProcessor` and `AutoModelForZeroShotObjectDetection`, moves the model to CPU or CUDA, and puts it in evaluation mode. The model is downloaded on first use and reused from the Hugging Face cache later. `metadata()` records the model name, revision, device, and the two inference thresholds.

`detect` validates that the image is RGB `uint8`, strips and lowercases the prompts, then asks the processor to encode the image and text as one batch. It runs the model under PyTorch `inference_mode()`, so gradients are not recorded. Finally it calls `post_process_grounded_object_detection` with `threshold=0.10`, `text_threshold=0.10`, `target_sizes=[image.shape[:2]]`, and `text_labels=[prompt_list]`. `target_sizes` supplies `(height, width)` so the returned boxes are scaled to the original image's pixel coordinates. The adapter converts each returned box, score, and text label into `Detection`, adding its model name. It does not use MuJoCo ground truth or apply a separate non-maximum suppression step.

There are two score decisions. DINO's box threshold controls **which candidates are returned and saved**. `evaluate.py`'s confidence threshold controls **which saved candidates are counted**. DINO saves candidates at score 0.10 or above, while evaluation counts boxes at score 0.35 or above. A box below 0.10 is unavailable in `result.json`; a saved box between 0.10 and 0.35 is inspectable but does not affect TP/FP/FN. To study lower-confidence proposals, lower the adapter's inference threshold and rerun into a new output directory. Merely lowering the evaluator threshold cannot recover boxes that were never saved.

## How inference becomes a result

`run.py` calls `render_scene(SCENE)` and gets `rgb`, `target_mask`, and `ground_truth`. It calls `detector.detect(rendered.rgb, [SCENE.prompt])` once. It then calls `evaluate(detections, ground_truth)`; the detector is not involved in this matching step. The runner saves RGB, the mask, and an annotation image, plus `result.json` containing the scene fields, prompts, ground truth, every returned detection, detector metadata, and evaluation.

`evaluate.py` sorts qualifying predictions by score. For a present target, the first prediction with the prompted label and IoU at least 0.50 with the visible-pixel ground-truth box is one true positive. Additional qualifying boxes, including duplicates or poorly placed boxes, are false positives. If no prediction matches a present target, it is one false negative. For an absent target, the ground-truth box is `null`; every qualifying prediction is a false positive and there is no false negative. An absent view with no detections has zero TP, FP, and FN; this program does not store a separate true-negative count. IoU is intersection area divided by union area of the two boxes.

The mask-derived box covers only **visible** target pixels. It can shrink under occlusion, while a model may predict a plausible whole-object box. Inspect `rgb.png`, `target_mask.png`, and `annotated.png` together before calling a low-IoU case a semantic model failure.

## Implement another detector

Create a module such as `my_detector.py` containing a class with `detect` and `metadata`. This is a template for adapting an actual model API, not a second detector shipped by this repository:

```python
from collections.abc import Sequence
from typing import Any

import numpy as np

from detector import Detection


class MyDetector:
    model_name = "provider/model-and-revision"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        # Load this model and its preprocessor once here.

    def detect(self, image: np.ndarray, prompts: Sequence[str]) -> list[Detection]:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected H x W x 3 RGB uint8 image")
        # Run the model with the supplied prompts. Convert each output to:
        # Detection(label=prompt_label, score=confidence,
        #           bbox=(left_px, top_px, right_px, bottom_px),
        #           model_name=self.model_name)
        # Return a list, including [] when the model finds nothing.
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "device": self.device}
```

Replace the `GroundingDINO` import and construction in `run.py:main` with `MyDetector`; `run_experiment`, rendering, matching, and saved JSON can remain the same. If your model returns normalized centre-format `(cx, cy, w, h)`, convert using image width `W` and height `H`: `left=(cx-w/2)*W`, `top=(cy-h/2)*H`, `right=(cx+w/2)*W`, `bottom=(cy+h/2)*H`. If it returns `(x, y, width, height)` in pixels, use `(x, y, x+width, y+height)`. Preserve the model's confidence ordering and map its class names to the prompted label explicitly. If it cannot answer a supplied prompt, reject that prompt rather than silently evaluating a different class.

Before trusting a new adapter, run one obvious present scene and one absent scene, inspect its boxes on `annotated.png`, and check that labels, scores, and coordinates in `result.json` match the model's output. Add a focused test of any coordinate or label conversion. Use a different output directory and record a distinct model name/revision so results from different adapters cannot be mistaken for the same run. This interface permits a comparison later; it does not imply that an additional adapter is already installed or validated.
