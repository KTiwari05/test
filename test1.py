import cv2
import os
import numpy as np
import logging
import time
from ultralytics import YOLO
from skimage import exposure

# ---------------------- Utility Kernel ---------------------- #
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# ---------------------- Order points utility ---------------------- #
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# ---------------------- Extract plate using YOLO bbox ---------------------- #
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
        crop = image[y1e:y2e, x1e:x2e]
        return crop
    except Exception as e:
        logging.error(f"Plate extraction failed: {e}")
        return None

# Define polygon coordinates
POLYGON = np.array([[726,466], [1255,474], [1403,1046], [600, 1057]], dtype=np.int32)

def apply_polygon_mask(frame):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [POLYGON], 255)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_frame

# ---------------------- FULL Preprocessing Pipeline ---------------------- #
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
    lab_image = cv2.merge([l, a, b])
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

def remove_white_pixels(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    non_white_mask = cv2.bitwise_not(white_mask)
    filtered_image = cv2.bitwise_and(image, image, mask=non_white_mask)
    kernel = np.ones((3,3), np.uint8)
    return cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, kernel)

def adaptive_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 5, 50, 50)
    thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 17, 9
    )
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cleaned

def sharpen_image(image):
    kernel = np.array([[0, -1,  0], [-1,  5, -1], [0, -1,  0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)

def fill_black_holes(binary_image):
    if len(binary_image.shape) == 3:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return closed

def full_preprocess_pipeline(plate_image):
    denoised = denoise_image(plate_image)
    gamma_corrected = adjust_gamma(denoised)
    enhanced = enhance_image_colors(gamma_corrected)
    filtered = remove_white_pixels(enhanced)
    thresholded = adaptive_thresholding(filtered)
    sharpened = sharpen_image(thresholded)
    final = fill_black_holes(sharpened)
    return final

def should_save_image(binary_image, min_white_ratio=0.01):
    if len(binary_image.shape) == 3:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    white_pixels = cv2.countNonZero(binary_image)
    total_pixels = binary_image.shape[0] * binary_image.shape[1]
    ratio = white_pixels / total_pixels
    return ratio > min_white_ratio

def quick_precheck(plate_crop, precheck_white_ratio=0.1):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixels = cv2.countNonZero(binary)
    total_pixels = binary.shape[0] * binary.shape[1]
    ratio = white_pixels / total_pixels
    return ratio > precheck_white_ratio

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    start_time = time.time()
    video_path = "test3plate.mp4"
    output_dir = "final_output1"
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO("best2.engine")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Unable to open video file.")
        return

    frame_count = 0
    saved_frame_count = 0
    frame_skip = 15

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        masked_frame = apply_polygon_mask(frame)
        results = model(masked_frame, verbose=False, device=0)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.cpu().int().tolist()
            for i, box in enumerate(boxes):
                if class_ids[i] == 1:
                    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                    if cv2.pointPolygonTest(POLYGON, (cx, cy), False) < 0:
                        continue
                    plate_crop = extract_plate(frame, box, margin=0.2)
                    if plate_crop is None:
                        continue
                    if not quick_precheck(plate_crop, precheck_white_ratio=0.01):
                        logging.info(f"Quick-skip frame {frame_count}_{i} due to low white pixel content in raw crop.")
                        continue
                    final_processed = full_preprocess_pipeline(plate_crop)
                    if should_save_image(final_processed, min_white_ratio=0.1):
                        frame_name = os.path.join(output_dir, f"plate_{frame_count:06d}_{i}.png")
                        if cv2.imwrite(frame_name, final_processed, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                            logging.info(f"Saved: {frame_name}")
                            saved_frame_count += 1
                    else:
                        logging.info(f"Skipped frame {frame_count}_{i} after full pipeline due to low white pixel content.")
        frame_count += 1

    cap.release()
    elapsed = time.time() - start_time
    logging.info(f"Completed! Saved {saved_frame_count} fully processed plates in {elapsed:.2f} seconds.")
    logging.info(f"Processed {frame_count} frames. Average time per frame: {elapsed / max(frame_count,1):.3f} sec")

if __name__ == "__main__":
    main()
