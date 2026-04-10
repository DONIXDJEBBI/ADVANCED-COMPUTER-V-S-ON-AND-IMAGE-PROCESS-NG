import os
import cv2
import pytesseract
import time
import gc
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from craft_text_detector import Craft


IMAGE_PATH = "images/img1.png"
OUT_DIR = "outputs"
USE_CUDA = False
PADDING = 15
MAX_BOXES = 40

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.makedirs(OUT_DIR, exist_ok=True)


try:
    import torch
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    model.eval()

    @torch.no_grad()
    def trocr_ocr(pil_img):
        pixel_values = processor(pil_img, return_tensors="pt").pixel_values
        ids = model.generate(pixel_values, num_beams=1, max_length=96)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
except ImportError:
    print("Warning: torch is not available. Skipping TrOCR pipeline.")
    trocr_ocr = None


craft = Craft(output_dir=None, crop_type="box", cuda=USE_CUDA)



def auto_rotate(image_bgr):
    try:
        osd = pytesseract.image_to_osd(image_bgr)
        for line in osd.split("\n"):
            if "Rotate" in line:
                angle = int(line.split(":")[-1])
                if angle == 90:
                    image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_180)
                elif angle == 270:
                    image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except:
        pass
    return image_bgr



def enhance_image(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(resized, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh


def tesseract_ocr(preprocessed_img):
    return pytesseract.image_to_string(preprocessed_img, lang="eng", config="--psm 6").strip()



def hybrid_ocr(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(image_path)

    image_bgr = auto_rotate(image_bgr)
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    W, H = pil_img.size
    texts = []

    try:
        result = craft.detect_text(image_path)
        boxes = result.get("boxes", [])[:MAX_BOXES]
    except:
        boxes = []

    for box in boxes:
        try:
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]

            x1 = max(min(xs) - PADDING, 0)
            y1 = max(min(ys) - PADDING, 0)
            x2 = min(max(xs) + PADDING, W)
            y2 = min(max(ys) + PADDING, H)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = pil_img.crop((x1, y1, x2, y2))
            if trocr_ocr is not None:
                text = trocr_ocr(crop)
                if len(text) > 2:
                    texts.append(text)
        except:
            continue

    trocr_text = " ".join(texts)

    if len(trocr_text.split()) < 4:
        print("CRAFT failed → fallback to Tesseract")
        enhanced = enhance_image(image_bgr)
        return tesseract_ocr(enhanced)

    print("Using CRAFT + TrOCR")
    return trocr_text


start = time.time()
final_text = hybrid_ocr(IMAGE_PATH)

with open(os.path.join(OUT_DIR, "result.txt"), "w", encoding="utf-8") as f:
    f.write(final_text)

print("\n===== FINAL RESULT =====")
print(final_text if final_text else "[EMPTY]")
print(f"\nTime: {time.time() - start:.2f} sec")

craft.unload_craftnet_model()
gc.collect()
def run_ocr(image_path):
    return hybrid_ocr(image_path)
