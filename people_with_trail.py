import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque

cap = cv2.VideoCapture("video_data/People_and_Cars_on_the_Road.mp4")
model = YOLO("yolov8n.pt")

id_map = {}                    # tracker obj_id -> small sequential ID (sid)
next_id = 0
trails = defaultdict(lambda: deque(maxlen=30))  # history of centers per obj_id
appear = defaultdict(int)      # how many frames we've seen this obj_id
colors = {}                    # sid -> BGR color

def get_color_for_sid(sid: int):
    # deterministic but distinct-ish color per sid
    rng = np.random.RandomState(sid * 9973 + 12345)
    return tuple(int(x) for x in rng.randint(60, 255, size=3))  # avoid very dark

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, classes=[0], persist=True, verbose=False)
    annotated = frame.copy()

    # Make sure we have detections with IDs
    if results and results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        # First pass: update IDs, trails
        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            appear[obj_id] += 1
            if appear[obj_id] >= 5 and obj_id not in id_map:
                # assign a compact sid and color once the track is stable
                id_map[obj_id] = next_id
                colors[next_id] = get_color_for_sid(next_id)
                next_id += 1

            # keep the center history even before we decide to show it
            trails[obj_id].append((cx, cy))

        # Second pass: draw boxes/labels for “adopted” tracks
        for box, obj_id in zip(boxes, ids):
            if obj_id not in id_map:
                continue
            sid = id_map[obj_id]
            color = colors[sid]
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f'ID: {sid}', (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

        # Finally: draw the trails (polylines) for adopted tracks
        for obj_id, pts in trails.items():
            if obj_id not in id_map or len(pts) < 2:
                continue
            sid = id_map[obj_id]
            color = colors[sid]
            # draw as short segments for robustness
            for i in range(1, len(pts)):
                cv2.line(annotated, pts[i-1], pts[i], color, 2)

    # show once per frame (after all drawing)
    cv2.imshow("Annotated Frame", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
