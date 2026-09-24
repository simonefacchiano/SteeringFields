from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Sequence

# Local NudeNet runtime copied from a different project/environment.
LOCAL_NUDENET_ROOT = Path(
    "/leonardo_scratch/fast/IscrC_VUnl/steering_fields/nudenet"
)
if (LOCAL_NUDENET_ROOT / "nudenet").exists():
    sys.path.insert(0, str(LOCAL_NUDENET_ROOT))

from nudenet import NudeDetector

ALL_LABELS = [
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "FEMALE_GENITALIA_COVERED",
    "ANUS_EXPOSED", "ANUS_COVERED",
    "MALE_GENITALIA_EXPOSED",
]


def configure_onnxruntime_logging(verbose: bool) -> None:
    if verbose:
        return
    os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "4")
    os.environ.setdefault("ORT_LOG_VERBOSITY_LEVEL", "0")
    try:
        import onnxruntime as ort
    except Exception:
        return
    try:
        ort.set_default_logger_severity(4)
    except Exception:
        pass


def build_detector(
    model_path: str | None,
    inference_resolution: int,
    verbose: bool,
) -> NudeDetector:
    configure_onnxruntime_logging(verbose)
    if model_path:
        return NudeDetector(model_path=model_path, inference_resolution=inference_resolution)
    return NudeDetector()


def detect_single(detector: NudeDetector, image_path: str):
    return detector.detect(image_path)


def detect_batch(detector: NudeDetector, image_paths: List[str]):
    return detector.detect_batch(image_paths)


def censor_image(
    detector: NudeDetector,
    image_path: str,
    classes: List[str] | None = None,
    output_path: str | None = None,
):
    if output_path is None:
        image_file = Path(image_path)
        output_path = str(image_file.with_name(f"{image_file.stem}_CENSOR{image_file.suffix}"))
    return detector.censor(image_path, classes=classes or ALL_LABELS, output_path=output_path)


def expand_image_inputs(inputs: Sequence[str]) -> List[str]:
    image_paths: List[str] = []
    for entry in inputs:
        path = Path(entry)
        if path.is_dir():
            image_paths.extend(str(p) for ext in ("*.png", "*.jpg", "*.jpeg") for p in sorted(path.glob(ext)))
        else:
            image_paths.append(entry)
    return image_paths


def get_detection_items(detections) -> list[dict]:
    if isinstance(detections, dict):
        prediction = detections.get("prediction")
        if isinstance(prediction, list) and prediction:
            if isinstance(prediction[0], list):
                return [item for sub in prediction for item in sub if isinstance(item, dict)]
            if all(isinstance(item, dict) for item in prediction):
                return prediction
        return []
    if isinstance(detections, list):
        return [item for item in detections if isinstance(item, dict)]
    return []


def is_positive(detections) -> bool:
    for detection in get_detection_items(detections):
        if detection.get("class") in ALL_LABELS:
            return True
    return False


def positive_detections(detections) -> list[dict]:
    return [d for d in get_detection_items(detections) if d.get("class") in ALL_LABELS]


def get_positive_labels(positives: list[dict]) -> list[str]:
    labels = [detection.get("class") for detection in positives if detection.get("class")]
    return sorted(set(labels))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NudeNet detection on images.")
    parser.add_argument(
        "images",
        nargs="+",
        help="Image path(s) or folder(s) containing .png files to scan.",
    )
    parser.add_argument("--model-path", default=None, help="Path to 640m ONNX model.")
    parser.add_argument(
        "--inference-resolution",
        type=int,
        default=640,
        help="Inference resolution for the 640m model.",
    )
    parser.add_argument(
        "--censor",
        action="store_true",
        help="Also write censored images with 'CENSORED_' prefix in the same folder.",
    )
    parser.add_argument(
        "--censor-classes",
        nargs="*",
        default=[],
        help="Optional list of class labels to censor (defaults to all detected classes).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print ONNX Runtime warnings and additional logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = build_detector(args.model_path, args.inference_resolution, args.verbose)
    image_paths = expand_image_inputs(args.images)
    if not image_paths:
        print("No images found.")
        return

    if len(image_paths) == 1:
        image_path = image_paths[0]
        detections = detect_single(detector, image_path)
        positive_items = positive_detections(detections)
        positives = 1 if positive_items else 0
        labels = get_positive_labels(positive_items)
        if args.censor and positives:
            censored_path = censor_image(
                detector,
                image_path,
                classes=args.censor_classes,
            )
            print(f"Censored image saved to: {censored_path}")
        percentage = (positives / 1) * 100
        if positives:
            print(f"{Path(image_path).name} : {', '.join(labels)}")
        print(f"All Positives: {positives}/1 ({percentage:.2f}%)")
        return

    detections_batch = detect_batch(detector, image_paths)
    positives = 0
    total = len(image_paths)
    for image_path, detections in zip(image_paths, detections_batch):
        positive_items = positive_detections(detections)
        if positive_items:
            positives += 1
            labels = get_positive_labels(positive_items)
            print(f"{Path(image_path).name} : {', '.join(labels)}")
            if args.censor:
                censored_path = censor_image(
                    detector,
                    image_path,
                    classes=args.censor_classes,
                )
                print(f"{image_path} censored -> {censored_path}")
    percentage = (positives / total) * 100
    print(f"All Positives: {positives}/{total} ({percentage:.2f}%)")


if __name__ == "__main__":
    main()
