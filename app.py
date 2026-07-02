import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR
import io
import json
import tempfile
import os
import time

# Initialize models
@st.cache_resource
def load_models():
    """Load YOLO and PaddleOCR models"""
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    return ocr

@st.cache_resource
def load_yolo_model(model_path):
    """Load YOLO model from uploaded file or path"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

def detect_plates(image, model, conf_threshold=0.25, roi=None):
    """Detect license plates using YOLO with optional ROI cropping"""
    plates = []
    
    # If ROI is specified, crop the image first
    if roi:
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        cropped_image = image[roi_y1:roi_y2, roi_x1:roi_x2]
        results = model(cropped_image, conf=conf_threshold)
        
        # Adjust bounding boxes to full image coordinates
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Convert coordinates back to full image space
                full_x1 = x1 + roi_x1
                full_y1 = y1 + roi_y1
                full_x2 = x2 + roi_x1
                full_y2 = y2 + roi_y1
                
                plates.append({
                    'bbox': (full_x1, full_y1, full_x2, full_y2),
                    'confidence': conf
                })
    else:
        # Process full image
        results = model(image, conf=conf_threshold)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                plates.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })
    
    return plates

def extract_text_from_plate(image, bbox, ocr, min_confidence=0.5, min_text_size=20):
    """Extract text from detected plate region with filtering"""
    x1, y1, x2, y2 = bbox
    plate_img = image[y1:y2, x1:x2]
    
    # Run OCR
    result = ocr.ocr(plate_img, cls=True)
    
    # Extract and filter text
    texts = []
    if result and result[0]:
        for line in result[0]:
            # line[0] contains box coordinates, line[1] contains (text, confidence)
            box_coords = line[0]
            text_content = line[1][0]
            confidence = line[1][1]
            
            # Calculate text box size
            box_height = max(box_coords, key=lambda x: x[1])[1] - min(box_coords, key=lambda x: x[1])[1]
            box_width = max(box_coords, key=lambda x: x[0])[0] - min(box_coords, key=lambda x: x[0])[0]
            text_size = max(box_height, box_width)
            
            # Filter by confidence and size
            if confidence >= min_confidence and text_size >= min_text_size:
                texts.append((text_content, confidence, text_size))
    
    # Sort by size (largest first) and return combined text
    texts.sort(key=lambda x: x[2], reverse=True)
    return " ".join([t[0] for t in texts])

def draw_results(image, plates, texts, roi=None):
    """Draw bounding boxes, text, and ROI on image"""
    img_copy = image.copy()
    
    # Draw ROI if specified
    if roi:
        x1, y1, x2, y2 = roi
        # Draw semi-transparent overlay outside ROI
        overlay = img_copy.copy()
        cv2.rectangle(overlay, (0, 0), (img_copy.shape[1], img_copy.shape[0]), (0, 0, 0), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img_copy, 0.5, 0, img_copy)
        
        # Draw ROI border
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.putText(img_copy, "ROI - Detection Area", (x1 + 10, y1 + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    
    for plate, text in zip(plates, texts):
        x1, y1, x2, y2 = plate['bbox']
        conf = plate['confidence']
        
        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label
        label = f"{text} ({conf:.2f})"
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_copy, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(img_copy, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return img_copy

def draw_roi_preview(image, roi):
    """Draw ROI preview on image without detection results"""
    img_copy = image.copy()
    
    if roi:
        x1, y1, x2, y2 = roi
        # Draw semi-transparent overlay outside ROI
        overlay = img_copy.copy()
        cv2.rectangle(overlay, (0, 0), (img_copy.shape[1], img_copy.shape[0]), (0, 0, 0), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img_copy, 0.5, 0, img_copy)
        
        # Draw ROI border
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 255, 0), 3)
        
        # Add text labels
        cv2.putText(img_copy, "ROI - Detection Area", (x1 + 10, y1 + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        cv2.putText(img_copy, f"({x1}, {y1})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img_copy, f"({x2}, {y2})", (x2 - 120, y2 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return img_copy

def process_and_display(img_bgr, display_image, yolo_model, ocr, conf_threshold, roi, use_roi,
                         min_ocr_confidence, min_text_size, col1, col2,
                         preview_caption="Original", key_prefix="frame"):
    """Run YOLO detection + OCR on a BGR frame and render results into col1/col2."""
    with col1:
        st.image(display_image, caption=preview_caption, use_container_width=True)

        if use_roi:
            st.info(f"📏 Image dimensions: {img_bgr.shape[1]} x {img_bgr.shape[0]} (width x height)")
            st.subheader("🎯 ROI Preview")
            roi_preview = draw_roi_preview(img_bgr, roi)
            roi_preview_rgb = cv2.cvtColor(roi_preview, cv2.COLOR_BGR2RGB)
            st.image(roi_preview_rgb, caption="ROI Visualization (Yellow = Detection Area)", use_container_width=True)

    with st.spinner("Detecting plates..."):
        plates = detect_plates(img_bgr, yolo_model, conf_threshold, roi)

    if not plates:
        st.warning("⚠️ No license plates detected. Try adjusting the confidence threshold.")
        return

    st.success(f"✅ Detected {len(plates)} license plate(s)")

    texts = []
    with st.spinner("Extracting text..."):
        for plate in plates:
            text = extract_text_from_plate(img_bgr, plate['bbox'], ocr, min_ocr_confidence, min_text_size)
            texts.append(text)

    result_img = draw_results(img_bgr, plates, texts, roi)
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("🎯 Detection Results")
        st.image(result_img_rgb, caption="Detected Plates", use_container_width=True)

    st.subheader("📝 Extracted Text")
    for i, (plate, text) in enumerate(zip(plates, texts), 1):
        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.metric(f"Plate {i}", f"{plate['confidence']:.2%}")
        with col_b:
            st.code(text if text else "No text detected", language=None)

    st.markdown("---")
    result_bytes = cv2.imencode('.jpg', result_img)[1].tobytes()
    st.download_button(
        label="📥 Download Result Image",
        data=result_bytes,
        file_name="plate_detection_result.jpg",
        mime="image/jpeg",
        key=f"download_{key_prefix}"
    )

def get_video_player_fragment(cap, frame_count, run_every):
    """Build a fragment that auto-advances and redraws just the video frame,
    without rerunning (and flickering) the rest of the page."""
    @st.fragment(run_every=run_every)
    def _video_player_fragment():
        frame_idx = min(st.session_state["vid_frame_idx"], frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()

        if not ret:
            st.error("Could not read frame from video.")
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption=f"Frame {frame_idx}/{frame_count - 1}", use_container_width=True)

        if st.session_state["vid_playing"]:
            if frame_idx < frame_count - 1:
                st.session_state["vid_frame_idx"] = frame_idx + 1
            else:
                st.session_state["vid_playing"] = False

    return _video_player_fragment

# Streamlit UI
st.set_page_config(page_title="License Plate OCR", layout="wide")
st.title("🚗 License Plate Detection & OCR")
st.markdown("Upload an image or video to detect license plates and extract text using YOLO + PaddleOCR")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # YOLO model path (fixed)
    model_path = st.text_input("YOLO Model Path:", value="Number_Plate.pt", 
                                help="Path to your YOLO model file (e.g., best.pt)")
    
    conf_threshold = st.slider("Confidence Threshold:", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown("### 🔤 OCR Filters")
    min_ocr_confidence = st.slider("Min OCR Confidence:", 0.0, 1.0, 0.5, 0.05,
                                    help="Ignore text with low confidence")
    min_text_size = st.slider("Min Text Size (pixels):", 0, 100, 20, 5,
                               help="Ignore small text (filters noise)")
    
    st.markdown("### 📐 ROI Settings")
    use_roi = st.checkbox("Enable ROI (Region of Interest)", value=False,
                          help="Detect plates only within a specific region")
    
    roi = None
    if use_roi:
        st.markdown("**Define ROI coordinates:**")
        col_a, col_b = st.columns(2)
        with col_a:
            roi_x1 = st.number_input("X1 (left):", min_value=0, value=0, step=10)
            roi_y1 = st.number_input("Y1 (top):", min_value=0, value=0, step=10)
        with col_b:
            roi_x2 = st.number_input("X2 (right):", min_value=0, value=640, step=10)
            roi_y2 = st.number_input("Y2 (bottom):", min_value=0, value=480, step=10)
        
        roi = (roi_x1, roi_y1, roi_x2, roi_y2)
        
        st.info("💡 Tip: Upload an image first, then adjust ROI coordinates to match the area you want to detect.")
    
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Set your YOLO model path (default: best.pt)
    2. Upload an image, or upload a video and capture a frame
    3. View detected plates and extracted text
    """)

# Load OCR model
ocr = load_models()

# Main content
tab_image, tab_video = st.tabs(["🖼️ Image", "🎥 Video"])

# ---- Image tab ----
with tab_image:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'], key="image_uploader")

    if uploaded_file and model_path:
        yolo_model = load_yolo_model(model_path)

        if yolo_model:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array

            process_and_display(img_bgr, image, yolo_model, ocr, conf_threshold, roi, use_roi,
                                 min_ocr_confidence, min_text_size, col1, col2,
                                 preview_caption="Original Image", key_prefix="image")

    elif uploaded_file and not model_path:
        st.warning("⚠️ Please upload or specify a YOLO model path first.")
    elif not uploaded_file:
        st.info("👆 Please upload an image to begin.")

# ---- Video tab ----
with tab_video:
    col1v, col2v = st.columns(2)

    with col1v:
        st.subheader("📤 Upload Video")
        video_file = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov', 'mkv'], key="video_uploader")

    if video_file and model_path:
        yolo_model = load_yolo_model(model_path)

        # Persist the upload to a temp file so OpenCV can read/seek it
        video_identity = f"{video_file.name}_{video_file.size}"
        if st.session_state.get("video_identity") != video_identity:
            old_path = st.session_state.get("video_temp_path")
            old_cap = st.session_state.get("vid_cap")
            if old_cap is not None:
                old_cap.release()
            if old_path and os.path.exists(old_path):
                os.remove(old_path)

            suffix = os.path.splitext(video_file.name)[1] or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(video_file.read())
            tmp.close()

            st.session_state["video_temp_path"] = tmp.name
            st.session_state["video_identity"] = video_identity
            st.session_state["vid_cap"] = cv2.VideoCapture(tmp.name)
            st.session_state["vid_playing"] = False
            st.session_state["vid_frame_idx"] = 0

        cap = st.session_state["vid_cap"]
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        if frame_count > 0 and yolo_model:
            with col1v:
                st.caption(f"{frame_count} frames · {fps:.1f} fps · {frame_count / fps:.1f}s")
                play_col, pause_col, restart_col = st.columns(3)
                with play_col:
                    if st.button("▶️ Play", use_container_width=True,
                                 disabled=st.session_state["vid_playing"]):
                        st.session_state["vid_playing"] = True
                with pause_col:
                    if st.button("⏸️ Pause", use_container_width=True,
                                 disabled=not st.session_state["vid_playing"]):
                        st.session_state["vid_playing"] = False
                with restart_col:
                    if st.button("⏮️ Restart", use_container_width=True):
                        st.session_state["vid_playing"] = False
                        st.session_state["vid_frame_idx"] = 0

                # Fragment only re-runs itself on each tick, so the sidebar,
                # tabs, and rest of the page never redraw -> no more blinking.
                run_every = (1 / fps) if st.session_state["vid_playing"] else None
                get_video_player_fragment(cap, frame_count, run_every)()

                capture_clicked = st.button("📸 Capture Frame & Detect", key="capture_frame_btn",
                                             use_container_width=True)

            if capture_clicked:
                st.session_state["vid_playing"] = False
                frame_idx = min(st.session_state["vid_frame_idx"], frame_count - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame_bgr = cap.read()

                if ret:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    process_and_display(frame_bgr, frame_rgb, yolo_model, ocr, conf_threshold, roi, use_roi,
                                         min_ocr_confidence, min_text_size, col1v, col2v,
                                         preview_caption=f"Captured Frame {frame_idx}", key_prefix="video")
                else:
                    st.error("Could not read the selected frame from the video.")
        else:
            st.error("Could not read frame count from the uploaded video.")

    elif video_file and not model_path:
        st.warning("⚠️ Please upload or specify a YOLO model path first.")
    elif not video_file:
        st.info("👆 Please upload a video to begin.")