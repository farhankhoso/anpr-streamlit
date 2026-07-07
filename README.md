# License Plate Detection & OCR

A Streamlit web app that combines a YOLO object detector with PaddleOCR text recognition to detect and read license plates from images and videos.

## Demo

🔗 **[Try it live here](https://anpr-web.streamlit.app/)**
## Features

### 🖼️ Image Tab
- Upload a JPG/PNG image
- YOLO detects plate bounding boxes
- Each cropped plate is passed through PaddleOCR
- Results (boxes, recognized text, confidence scores) are drawn on the image and listed below
- Download button for the annotated result

### 🎥 Video Tab
- Upload a video — it's saved to a temp file and opened with OpenCV
- A custom frame-by-frame player (`st.fragment(run_every=1/fps)`) plays the video at its real source FPS
- Only the player widget updates on each tick, not the whole page, so there's no flicker
- **Play / Pause / Restart** controls for playback
- **📸 Capture Frame & Detect** grabs the currently displayed frame and runs it through the same YOLO → OCR pipeline as the image tab

### Sidebar Controls
- YOLO model path (editable)
- Detection confidence threshold (default `0.5`)
- OCR confidence / text-size filters
- Optional ROI (region of interest) to restrict detection to part of the frame

## Project Structure / Key Functions

| Function | Purpose |
|---|---|
| `load_models()` / `load_yolo_model()` | Cached model loaders (`@st.cache_resource`) so PaddleOCR/YOLO aren't reloaded on every interaction |
| `detect_plates()` | Runs YOLO detection on a frame/image |
| `extract_text_from_plate()` | Runs PaddleOCR on a cropped plate region |
| `draw_results()` | Draws detection boxes + OCR text/confidence on the frame |
| `draw_roi_preview()` | Visualizes the configured ROI |
| `process_and_display()` | Shared detect → OCR → draw → show → download pipeline used by both the image tab and video capture flow |
| `get_video_player_fragment()` | Builds the fragment-scoped auto-advancing frame player |

## Requirements

Python **3.10** is required (stable combination for `paddleocr` + `ultralytics`). Dependencies are listed in `requirements.txt`.

## Setup & Run

1. Clone the repo:
   ```bash
   git clone https://github.com/farhankhoso/anpr-streamlit.git
   cd 'anpr-streamlit'
   ```

2. Create a virtual environment (Python 3.10):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Make sure the YOLO weights file `Number_Plate.pt` is present in the project directory** (same folder as `app.py`). The app defaults to this filename, but the path is editable from the sidebar if yours is named/located differently.

5. Run the app:
   ```bash
   streamlit run app.py
   ```

6. Open your browser at [http://localhost:8501](http://localhost:8501)

## Repo Checklist

Before pushing / running, make sure these are present in the project root:
- `app.py`
- `Number_Plate.pt` (trained YOLO weights)
- `requirements.txt`

## Troubleshooting

**`OSError: [WinError 127] The specified procedure could not be found` (torch/paddle DLL conflict on Windows)**
This happens when PyTorch and PaddlePaddle DLLs (OpenMP/MKL) collide in the same process. Add this to the top of your entry script before any other imports:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

## License

Add your license of choice here (e.g. MIT).
