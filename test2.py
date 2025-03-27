import cv2
import numpy as np
import re
import os
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from ctc_decoder import ctc_greedy_decoder  # You need to implement or import this

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

CHARACTER_SET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~  '
CHARACTER_DICT = {i: c for i, c in enumerate(CHARACTER_SET)}

# TRT helper
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    return inputs, outputs, bindings, stream

def filter_number(text):
    digits = re.sub(r'\D', '', text)
    if digits:
        number = int(digits)
        if 1 <= number <= 100:
            return f"{number:02d}"
    return None

def enhance_contrast_gpu(image_gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(image_gray)

def deskew_image(image_gray):
    coords = np.column_stack(np.where(image_gray > 0))
    if coords.shape[0] < 10:
        return image_gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = image_gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def recognize_image_trt(image_path, engine, context, inputs, outputs, bindings, stream):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    enhanced = enhance_contrast_gpu(image)
    deskewed = deskew_image(enhanced)
    resized = cv2.resize(deskewed, (100, 32))
    normalized = resized.astype(np.float32) / 255.0
    input_tensor = normalized[np.newaxis, np.newaxis, :, :]  # [1,1,32,100]
    np.copyto(inputs[0][0], input_tensor.ravel())

    # Inference
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()

    output_array = outputs[0][0].reshape((26, len(CHARACTER_SET) + 1))  # T x C
    decoded_text = ctc_greedy_decoder(output_array, CHARACTER_DICT)
    return filter_number(decoded_text)

def main():
    start_time = time.time()

    folder_path = "final_output"
    valid_exts = ('.png', '.jpg', '.jpeg')
    engine_path = "english_g2.trt"

    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    count = 0
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(valid_exts):
            continue
        image_path = os.path.join(folder_path, filename)
        recognized = recognize_image_trt(image_path, engine, context, inputs, outputs, bindings, stream)
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