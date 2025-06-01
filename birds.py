import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8s.pt")
video_path = "D://9.mp4"
cap = cv2.VideoCapture(video_path)
tracker = DeepSort(max_age=30)
counted_ids = set()


target_width, target_height = 640, 360

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    frame = cv2.resize(frame, (target_width, target_height))

    results = model(frame)[0]

    detections = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == 14 and conf > 0.75:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf.item(), 'bird'))

    tracks = tracker.update_tracks(detections, frame=frame)

    bird_count = 0
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        if track_id not in counted_ids:
            counted_ids.add(track_id)
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Bird {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        bird_count += 1

    cv2.putText(frame, f"Birds Now: {bird_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Bird Detection (Resized)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

