import cv2
import torch
import numpy as np
import re
import os
import time
from easyocr.recognition import get_recognizer, get_text

def filter_number(text):
    digits = re.sub(r'\D', '', text)
    if digits:
        number = int(digits)
        if 1 <= number <= 100:
            return f"{number:02d}"
    return None

# GPU-optimized contrast enhancement using CLAHE
def enhance_contrast_gpu(image_gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(image_gray)

# Efficient deskewing
def deskew_image(image_gray):
    coords = np.column_stack(np.where(image_gray > 0))
    if coords.shape[0] < 10:
        return image_gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = image_gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Main inference function
def recognize_image(image_path, recognizer, converter, character, device):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    enhanced = enhance_contrast_gpu(image)
    deskewed = deskew_image(enhanced)

    image_tensor = torch.from_numpy(deskewed).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

    image_list = [([0, 0, deskewed.shape[1], deskewed.shape[0]], deskewed)]

    results = get_text(
        character, 32, 100, recognizer, converter, image_list,
        decoder='greedy', beamWidth=1, batch_size=1,
        contrast_ths=0.05, adjust_contrast=0.7,
        filter_ths=0.003, workers=0, device=device
    )

    for _, text, _ in results:
        return filter_number(text)
    return None

def main():
    start_time = time.time()

    folder_path = "final_output"
    valid_exts = ('.png', '.jpg', '.jpeg')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    recog_model_path = "english_g2.pth"
    recog_network = 'generation2'
    network_params = {"input_channel": 1, "output_channel": 256, "hidden_size": 256}
    character = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~  '

    recognizer, converter = get_recognizer(
        recog_network, network_params, character,
        {}, {}, recog_model_path, device=device
    )

    count = 0
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(valid_exts):
            continue
        image_path = os.path.join(folder_path, filename)
        recognized = recognize_image(image_path, recognizer, converter, character, device)
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