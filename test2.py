import cv2
import numpy as np
import re
import os
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from multiprocessing import Pool, cpu_count

# ----------------- CTC Decoder -----------------
def ctc_greedy_decoder(output, char_dict, blank=0):
    pred_indices = np.argmax(output, axis=1)
    prev_idx = blank
    decoded = []
    for idx in pred_indices:
        if idx != blank and idx != prev_idx:
            decoded.append(char_dict.get(idx, ''))
        prev_idx = idx
    return ''.join(decoded)

# ----------------- Preprocessing -----------------
def filter_number(text):
    digits = re.sub(r'\D', '', text)
    if not digits:
        return None
    try:
        number = int(digits)
        if 1 <= number <= 100:
            return f"{number:02d}"
    except ValueError:
        return None
    return None

def enhance_contrast(image_gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(image_gray)

def deskew_image(image_gray):
    coords = np.column_stack(np.where(image_gray > 0))
    if coords.shape[0] < 10:
        return image_gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = image_gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return (os.path.basename(image_path), None)
    enhanced = enhance_contrast(image)
    deskewed = deskew_image(enhanced)
    resized = cv2.resize(deskewed, (100, 32))
    norm_img = resized.astype(np.float32) / 255.0
    norm_img = norm_img[np.newaxis, np.newaxis, :, :]
    return (os.path.basename(image_path), norm_img)

# ----------------- TRT Engine Wrapper -----------------
class TRTRecognizer:
    def __init__(self, engine_path, character):
        self.character = character
        self.char_dict = {i + 1: c for i, c in enumerate(character)}
        self.char_dict[0] = ''
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = (1, 1, 32, 100)
        self.context.set_input_shape(self.input_name, self.input_shape)
        self.output_shape = self.context.get_tensor_shape(self.output_name)
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
        self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * np.dtype(self.input_dtype).itemsize)
        self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * np.dtype(self.output_dtype).itemsize)
        self.bindings = [int(self.d_input), int(self.d_output)]

    def infer(self, image_tensor):
        np_input = image_tensor.astype(self.input_dtype)
        np_output = np.empty(self.output_shape, dtype=self.output_dtype)
        cuda.memcpy_htod(self.d_input, np_input)
        self.context.execute_v2(self.bindings)
        cuda.memcpy_dtoh(np_output, self.d_output)
        if np_output.ndim == 3:
            np_output = np_output[0]
        text = ctc_greedy_decoder(np_output, self.char_dict)
        return filter_number(text)

# ----------------- Main -----------------
def main():
    start_time = time.time()

    folder_path = r"final_output"
    engine_path = r"crnn_int8.trt"
    valid_exts = ('.png', '.jpg', '.jpeg')
    character = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~  '
    recognizer = TRTRecognizer(engine_path, character)

    # Warm-up
    dummy = np.zeros((1, 1, 32, 100), dtype=np.float32)
    recognizer.infer(dummy)

    # Load all image paths
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]

    # Parallel preprocessing
    with Pool(processes=min(4, cpu_count())) as pool:
        image_data = pool.map(preprocess_image, image_files)

    count = 0
    for filename, image_tensor in image_data:
        if image_tensor is None:
            print(f"{filename} -> [Invalid image]")
            continue
        recognized = recognizer.infer(image_tensor)
        if recognized:
            print(f"{filename} -> {recognized}")
        else:
            print(f"{filename} -> [No valid number]")
        count += 1

    elapsed = time.time() - start_time
    print(f"\nProcessed {count} images in {elapsed:.2f} seconds.")
    print(f"Average time per image: {elapsed / count:.2f} sec")

if __name__ == "__main__":
    main()
