import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n-seg.pt")
cap = cv2.VideoCapture("video_data/People_and_Cars_on_the_Road.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(source=frame, classes=[0], persist=True, verbose=False)
    annotated = frame.copy()

    # results can be list-like; normalize to a list
    batch = results if isinstance(results, (list, tuple)) else [results]

    for r in batch:
        if r.masks is None or r.boxes is None or r.boxes.id is None:
            continue

        masks = r.masks.data.cpu().numpy()         # (N,h,w) floats [0..1]
        boxes = r.boxes.xyxy.cpu().numpy()
        ids   = r.boxes.id.cpu().numpy().astype(int)

        n = min(len(masks), len(boxes), len(ids))
        H, W = frame.shape[:2]

        for i in range(n):
            m = masks[i]
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)  # keep binary edges
            bin_mask = m > 0.5                                          # boolean mask

            # --- FIX: blend with a color only on masked pixels ---
            color = np.array([0, 255, 0], dtype=np.float32)  # BGR
            region = annotated[bin_mask].astype(np.float32)
            region = 0.3 * region + 0.7 * color             # blend per-pixel
            annotated[bin_mask] = region.astype(np.uint8)

            # Contours (need 0/1 or 0/255 8-bit)
            cnts, _ = cv2.findContours(bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, cnts, -1, (0, 255, 0), 2)

            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated, f'ID:{ids[i]}', (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Segmentation", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
