import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_CONFIDENCE = 0.6


def get_default_model_path() -> str:
    repo_root = Path(__file__).resolve().parent
    candidate = repo_root / "model" / "best.pt"
    return str(candidate)


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def load_model(model_path: str) -> YOLO:
    logger = logging.getLogger("weapon_detection")
    resolved_path = Path(model_path).expanduser().resolve()
    if not resolved_path.exists():
        logger.error("Model file not found: %s", resolved_path)
        raise FileNotFoundError(f"Model file not found: {resolved_path}")
    try:
        model = YOLO(str(resolved_path))
        logger.info("Loaded YOLO model from %s", resolved_path)
        return model
    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)
        raise


def ensure_output_path(output: str | None, fallback_name: str, default_dir: str | None = None) -> str | None:
    if output is None:
        return None if default_dir is None else str(Path(default_dir).expanduser().resolve() / fallback_name)
    output_path = Path(output).expanduser().resolve()
    if output_path.is_dir():
        output_path = output_path / fallback_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def annotate_frame_with_model(model: YOLO, frame: np.ndarray, conf_threshold: float) -> np.ndarray:
    results = model(frame, conf=conf_threshold, verbose=False)
    return results[0].plot()


def detect_from_webcam(model: YOLO, output_path: str | None, conf_threshold: float = DEFAULT_CONFIDENCE, show_window: bool = True) -> str | None:
    logger = logging.getLogger("weapon_detection.webcam")
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        logger.error("Could not open webcam (index 0)")
        return None

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    if output_path is not None:
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                logger.warning("Failed to grab frame from webcam")
                break

            annotated = annotate_frame_with_model(model, frame, conf_threshold)

            if show_window:
                cv2.imshow("Webcam Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer is not None:
                writer.write(annotated)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()

    return output_path


def detect_in_video(model: YOLO, input_path: str, output_path: str, conf_threshold: float = DEFAULT_CONFIDENCE, show_window: bool = False) -> str:
    logger = logging.getLogger("weapon_detection.video")
    cap_path = Path(input_path).expanduser().resolve()
    if not cap_path.exists():
        raise FileNotFoundError(f"Video file not found: {cap_path}")

    capture = cv2.VideoCapture(str(cap_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {cap_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer_path = Path(output_path).expanduser().resolve()
    writer_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(writer_path), fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            annotated = annotate_frame_with_model(model, frame, conf_threshold)
            if show_window:
                cv2.imshow("Video Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            writer.write(annotated)
    finally:
        capture.release()
        writer.release()
        if show_window:
            cv2.destroyAllWindows()

    logger.info("Saved annotated video to %s", writer_path)
    return str(writer_path)


def detect_in_image(model: YOLO, input_path: str, output_path: str | None, conf_threshold: float = DEFAULT_CONFIDENCE, show_window: bool = False) -> str:
    logger = logging.getLogger("weapon_detection.image")
    img_path = Path(input_path).expanduser().resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    image = cv2.imread(str(img_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    annotated = annotate_frame_with_model(model, image, conf_threshold)

    out_path = ensure_output_path(output_path, f"{img_path.stem}_detected{img_path.suffix}", default_dir=str(img_path.parent))
    assert out_path is not None
    cv2.imwrite(out_path, annotated)
    logger.info("Saved annotated image to %s", out_path)
    if show_window:
        cv2.imshow("Image Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weapon-detection",
        description="Detect weapons from image, video, or webcam using a YOLO model",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=get_default_model_path(),
        help="Path to the YOLO model (.pt)",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Confidence threshold (default: {DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Image
    p_img = subparsers.add_parser("image", help="Run detection on a single image")
    p_img.add_argument("--input", required=True, type=str, help="Path to input image")
    p_img.add_argument("--output", required=False, type=str, help="Path or directory for output image")
    p_img.add_argument("--show", action="store_true", help="Show a window with the result")

    # Video
    p_vid = subparsers.add_parser("video", help="Run detection on a video file")
    p_vid.add_argument("--input", required=True, type=str, help="Path to input video")
    p_vid.add_argument("--output", required=True, type=str, help="Path or directory for output video")
    p_vid.add_argument("--show", action="store_true", help="Show a window with the result while processing")

    # Webcam
    p_cam = subparsers.add_parser("webcam", help="Run detection from webcam")
    p_cam.add_argument("--output", required=False, type=str, help="Path or directory to save the captured video (optional)")
    p_cam.add_argument("--show", action="store_true", help="Show a window with the result (default true if omitted)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    logger = logging.getLogger("weapon_detection")

    # Load model
    try:
        model = load_model(args.model_path)
    except Exception as exc:
        logger.error("Exiting due to model load failure: %s", exc)
        return 2

    try:
        if args.command == "image":
            out = ensure_output_path(args.output, f"{Path(args.input).stem}_detected{Path(args.input).suffix}")
            result = detect_in_image(model, args.input, out, conf_threshold=args.conf_threshold, show_window=args.show)
            logger.info("Image processing complete: %s", result)
        elif args.command == "video":
            out = ensure_output_path(args.output, f"{Path(args.input).stem}_detected.mp4")
            assert out is not None
            result = detect_in_video(model, args.input, out, conf_threshold=args.conf_threshold, show_window=args.show)
            logger.info("Video processing complete: %s", result)
        elif args.command == "webcam":
            out = ensure_output_path(args.output, "webcam_detected.mp4") if getattr(args, "output", None) else None
            result = detect_from_webcam(model, out, conf_threshold=args.conf_threshold, show_window=True if args.show else True)
            logger.info("Webcam processing complete%s", f": {result}" if result else "")
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Unhandled error during processing")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())