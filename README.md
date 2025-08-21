## Weapon Detection (YOLO)

Production-ready CLI to run a trained YOLO model on images, videos, or a live webcam and save annotated outputs. Default confidence threshold is 0.4.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place your model at `WeaponDetection/model/best.pt` or pass `--model_path`.

### Usage

Global flags:
- `--model_path PATH` path to `.pt` weights (default: `WeaponDetection/model/best.pt`)
- `--conf_threshold FLOAT` confidence threshold (default: 0.4)
- `-v` or `-vv` for more logs

Run on an image:

```bash
python main.py image --input path/to/image.jpg --output outputs/
```

Run on a video:

```bash
python main.py video --input path/to/video.mp4 --output outputs/video_out.mp4
```

Run from webcam (press `q` to quit):

```bash
python main.py webcam --output outputs/webcam_out.mp4
```

Show windows while processing (optional):

```bash
python main.py video --input in.mp4 --output out.mp4 --show -v
```

### Notes
- Outputs are auto-created; if `--output` is a directory, filenames are derived.
- Model and large outputs are git-ignored.

