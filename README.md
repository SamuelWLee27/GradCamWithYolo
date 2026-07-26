# GradCAM with YOLO

Visualize where a yolo detector "looks" when it makes a prediction. The script runs two class activation map methods on an image and saves the heatmaps:

- **EigenCAM**: class-agnostic, highlights the most salient regions overall (tracks activation variance).
- **Grad-CAM**: targeted at a specific class (COCO `dog` by default), highlights the pixels that raised that class's score.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From the **project root**:

```bash
python3 -m scripts.run
```

Run it as a module (`-m scripts.run`) rather than `python3 scripts/run.py`. The module form puts the project root on the import path, so the `src` package resolves. Running the file directly fails with `ModuleNotFoundError: No module named 'src'`.

On first run it downloads `yolov8n.pt`, reads `data/dog.jpeg`, and writes:

- `result/eigen_cam.jpg`
- `result/grad_cam.jpg`

To use a different image or target a different COCO class, edit the paths and the `DOG` class index at the top of `scripts/run.py`.

## Structure

```
scripts/run.py     # Entry point
src/image/         # Resize and crop preprocessing
src/yolo/          # Model loading plus wrappers
src/grad_cam/      # EigenCAM plus GradCAM
data/              # Input images
result/            # Output heatmaps
```