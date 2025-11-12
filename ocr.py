import threading
import time
from typing import List, Tuple

import cv2
import easyocr
import numpy as np
import supervision as sv
from fastapi import FastAPI, HTTPException
from inference_sdk import InferenceHTTPClient

MODEL_ID = "license-plate-recognition-rxg4e/11"
WINDOW_NAME = "Deteccao de Placas"
DETECTION_INTERVAL_SECONDS = 0.5

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="wsNv1Yo4iSAU0CuXa0mI",
)

BOX_ANNOTATOR = sv.BoxAnnotator()
EASY_OCR = easyocr.Reader(["en"], gpu=False)
CAMERA_LOCK = threading.Lock()

app = FastAPI(title="Car Plate OCR")


def _preprocess_plate(plate: np.ndarray) -> np.ndarray:
    resized = cv2.resize(plate, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.equalizeHist(blur)


def _read_plate_text(plate: np.ndarray) -> str:
    if plate.size == 0:
        return ""
    processed = _preprocess_plate(plate)
    results = EASY_OCR.readtext(processed)
    if not results:
        return ""
    raw_text = "".join(text for (_, text, _) in results)
    cleaned = "".join(char for char in raw_text if char.isalnum()).upper()
    return cleaned


def _annotate_frame(frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    inference = CLIENT.infer(frame, model_id=MODEL_ID)
    detections = sv.Detections.from_inference(inference)
    labels: List[str] = []
    detected_texts: List[str] = []

    for xyxy in detections.xyxy:
        x_min, y_min, x_max, y_max = map(int, xyxy)
        plate = frame[y_min:y_max, x_min:x_max]
        text = _read_plate_text(plate)
        label = text if text else "Sem leitura"
        labels.append(label)
        if text:
            detected_texts.append(text)

    annotated = frame.copy()
    if len(detections) > 0:
        annotated = BOX_ANNOTATOR.annotate(scene=annotated, detections=detections)
        for xyxy, label in zip(detections.xyxy, labels):
            if not label or label == "Sem leitura":
                continue
            x_min, y_min, _, _ = map(int, xyxy)
            cv2.putText(
                annotated,
                label,
                (x_min, max(y_min - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
    return annotated, detected_texts


def _capture_from_camera() -> str:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Nao foi possivel acessar a camera do dispositivo.")

    detected_plate = ""
    last_detection_ts = 0.0
    annotated_frame = None

    try:
        while not detected_plate:
            ok, frame = camera.read()
            if not ok:
                continue
            now = time.time()
            if (
                annotated_frame is None
                or now - last_detection_ts >= DETECTION_INTERVAL_SECONDS
            ):
                annotated_frame, texts = _annotate_frame(frame)
                last_detection_ts = now
                if texts:
                    detected_plate = texts[0]

            display = annotated_frame if annotated_frame is not None else frame
            cv2.imshow(WINDOW_NAME, display)
            cv2.waitKey(1)
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not detected_plate:
        raise RuntimeError("Nenhuma placa foi reconhecida. Tente novamente.")
    return detected_plate


@app.api_route("/plates", methods=["GET", "POST"])
def read_plates() -> dict:
    if not CAMERA_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A camera ja esta em uso. Tente novamente em instantes.",
        )

    try:
        plate = _capture_from_camera()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        CAMERA_LOCK.release()

    return {"plate": plate}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ocr:app", host="0.0.0.0", port=8000, reload=False)
