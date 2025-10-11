from inference_sdk import InferenceHTTPClient
import supervision as sv
import cv2
import os
import numpy as np
import easyocr

# Detector de placas adaptado de https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/model/11
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="wsNv1Yo4iSAU0CuXa0mI"
)

result = CLIENT.infer("placabrasil.jpg", model_id="license-plate-recognition-rxg4e/11")
detections = sv.Detections.from_inference(result)
image = cv2.imread("placabrasil.jpg")

annotator = sv.BoxAnnotator()
annotated_image = annotator.annotate(scene=image.copy(), detections=detections)
sv.plot_image(annotated_image)

os.makedirs("img_placas", exist_ok=True)

for i in range(len(detections)):
    x_min, y_min, x_max, y_max = map(int, detections.xyxy[i])
    plate_img = image[y_min:y_max, x_min:x_max]

    if plate_img.size == 0:
        print(f"Placa {i+1} vazia, ignorada.")
        continue

    crop_path = f"img_placas/placa_{i+1}.jpg"
    cv2.imwrite(crop_path, plate_img)
    print(f"Placa {i+1} salva em: {crop_path}")


# Extrator de caracteres adaptado de https://github.com/JaidedAI/EasyOCR

reader = easyocr.Reader(['en'], gpu=False) 

print("\n🔍 Iniciando OCR em placas recortadas (EasyOCR)...")

for img_file in os.listdir("img_placas"):
    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join("img_placas", img_file)
    plate = cv2.imread(img_path)

    plate = cv2.resize(plate, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)

    # OCR com EasyOCR
    results = reader.readtext(gray)

    if results:
        detected_texts = []
        for (bbox, text, conf) in results:
            detected_texts.append(text)
        text = " ".join(detected_texts)
    else:
        text = "[nenhum texto lido]"

    print(f"{img_file} - Texto detectado: {text}")

    cv2.putText(plate, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(f"img_placas/ocr_{img_file}", plate)
