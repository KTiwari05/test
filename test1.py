import cv2
import os
import numpy as np
import logging
import time
import argparse
from ultralytics import YOLO
from skimage import exposure

# ---------------------- Constants & Kernel ---------------------- #
POLYGON = np.array([[726,466], [1255,474], [1403,1046], [600, 1057]], dtype=np.int32)
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# ---------------------- Helper Functions ---------------------- #
def extract_plate(image, box, margin=0.2):
    try:
        x1, y1, x2, y2 = box
        h, w = image.shape[:2]
        box_w, box_h = x2 - x1, y2 - y1
        aspect_ratio = box_w / box_h if box_h != 0 else 0
        if 2.0 < aspect_ratio < 6.0:
            margin *= 1.2
        dx = int(margin * box_w)
        dy = int(margin * box_h)
        x1e, y1e = max(0, x1 - dx), max(0, y1 - dy)
        x2e, y2e = min(w, x2 + dx), min(h, y2 + dy)
        return image[y1e:y2e, x1e:x2e]
    except Exception as e:
        logging.error(f"Plate extraction failed: {e}")
        return None

def apply_polygon_mask(frame):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [POLYGON], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)

def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def adjust_gamma(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    gamma = 0.8 if brightness < 100 else 1.2 if brightness > 150 else 1.0
    return exposure.adjust_gamma(image, gamma)

def enhance_image_colors(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    a = cv2.equalizeHist(a)
    b = cv2.equalizeHist(b)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def remove_white_pixels(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    return cv2.bitwise_and(image, image, mask=cv2.bitwise_not(white_mask))

def adaptive_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 5, 50, 50)
    thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 17, 9
    )
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, MORPH_KERNEL)

def sharpen_image(image):
    kernel = np.array([[0, -1,  0], [-1,  5, -1], [0, -1,  0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)

def fill_black_holes(binary_image):
    if len(binary_image.shape) == 3:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, MORPH_KERNEL)

def full_preprocess_pipeline(image):
    for step in [
        denoise_image,
        adjust_gamma,
        enhance_image_colors,
        remove_white_pixels,
        adaptive_thresholding,
        sharpen_image,
        fill_black_holes
    ]:
        image = step(image)
    return image

def should_save_image(binary_image, min_white_ratio=0.01):
    if len(binary_image.shape) == 3:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    white_pixels = cv2.countNonZero(binary_image)
    total_pixels = binary_image.size
    return (white_pixels / total_pixels) > min_white_ratio

def quick_precheck(image, threshold=0.01):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (cv2.countNonZero(binary) / binary.size) > threshold

# ---------------------- Main ---------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp_url", type=str, default="", help="RTSP URL or leave blank for file mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    output_dir = "final_output1"
    os.makedirs(output_dir, exist_ok=True)

    # ---- Video Capture Init ---- #
    if args.rtsp_url.strip() != "":
        gst_pipeline = (
            f"rtspsrc location={args.rtsp_url} ! "
            "rtph264depay ! decodebin ! videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=1"
        )
        print(f"RTSP URL: {args.rtsp_url}")
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("Warning: Unable to open RTSP stream. Falling back to local 'test3plate.mp4'")
            cap = cv2.VideoCapture("test3plate.mp4")
    else:
        print("No RTSP URL provided. Using local file: test3plate.mp4")
        cap = cv2.VideoCapture("test3plate.mp4")

    if not cap.isOpened():
        logging.error("Unable to open video source.")
        return

    model = YOLO("best2.engine")  # Ensure this is Jetson-compatible

    frame_count = 0
    saved_frame_count = 0
    frame_skip = 15
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        masked = apply_polygon_mask(frame)
        results = model(masked, verbose=False, device=0)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.cpu().int().tolist()
            for i, box in enumerate(boxes):
                if class_ids[i] != 1:
                    continue
                cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                if cv2.pointPolygonTest(POLYGON, (cx, cy), False) < 0:
                    continue
                plate_crop = extract_plate(frame, box)
                if plate_crop is None or not quick_precheck(plate_crop):
                    continue
                processed = full_preprocess_pipeline(plate_crop)
                if should_save_image(processed, min_white_ratio=0.1):
                    fname = os.path.join(output_dir, f"plate_{frame_count:06d}_{i}.png")
                    if cv2.imwrite(fname, processed, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                        logging.info(f"Saved: {fname}")
                        saved_frame_count += 1
                else:
                    logging.info(f"Skipped {frame_count}_{i} after full pipeline")

        frame_count += 1

    cap.release()
    elapsed = time.time() - start_time
    logging.info(f"✅ Completed: {saved_frame_count} plates in {elapsed:.2f}s")
    logging.info(f"📦 Frames processed: {frame_count}, Avg/frame: {elapsed / max(1, frame_count):.3f}s")

if __name__ == "__main__":
    main()
