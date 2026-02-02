import cv2
import numpy as np

# ---------------- CONFIG ----------------
REFERENCE_IMAGE_PATH = "data/reference.jpg"
VIDEO_PATH = "demo/demo.mp4"

USE_CAMERA = False      # True → webcam, False → video file
CAMERA_INDEX = 0

SCALE = 0.6

if USE_CAMERA:
    MIN_DETECT_MATCHES = 12
    MIN_INLIERS = 8
else:
    MIN_DETECT_MATCHES = 12
    MIN_INLIERS = 8

DETECTION_INTERVAL = 8
MIN_CONFIRM_MATCHES = 6

MIN_TRACKER_SIZE = 25
# ---------------------------------------


def sanitize_bbox(x, y, w, h, frame_w, frame_h):
    x = int(max(0, x))
    y = int(max(0, y))
    w = int(min(w, frame_w - x))
    h = int(min(h, frame_h - y))

    if w <= MIN_TRACKER_SIZE or h <= MIN_TRACKER_SIZE:
        return None
    return (x, y, w, h)


# ---------- Load Reference ----------
ref_img = cv2.imread(REFERENCE_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if ref_img is None:
    raise FileNotFoundError("Reference image not found")

h_ref, w_ref = ref_img.shape

# ---------- SIFT ----------
sift = cv2.SIFT_create(
    nfeatures=600,
    contrastThreshold=0.01
)
kp_ref, des_ref = sift.detectAndCompute(ref_img, None)
if des_ref is None:
    raise RuntimeError("No features found in reference image")

bf = cv2.BFMatcher(cv2.NORM_L2)

# ---------- Video / Camera ----------
if USE_CAMERA:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
else:
    cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Cannot open video or camera")

tracker = None
tracking = False
frame_count = 0

print("[INFO] Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if USE_CAMERA:
        frame = cv2.flip(frame, 1)

    frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
    frame_h, frame_w = frame.shape[:2]

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if USE_CAMERA:
        gray_frame = cv2.equalizeHist(gray_frame)

    frame_count += 1


    # TRACKING MODE
    if tracking and tracker is not None:
        success, box = tracker.update(frame)
        if not success:
            tracker = None
            tracking = False
            continue

        x, y, w, h = map(int, box)

        bbox = sanitize_bbox(x, y, w, h, frame_w, frame_h)
        if bbox is None:
            tracker = None
            tracking = False
            continue

        x, y, w, h = bbox

        # ---------- Periodic Validation ----------
        if frame_count % DETECTION_INTERVAL == 0:
            roi = gray_frame[y:y+h, x:x+w]
            if roi.size == 0:
                tracker = None
                tracking = False
                continue

            kp_roi, des_roi = sift.detectAndCompute(roi, None)
            if des_roi is None:
                tracker = None
                tracking = False
                continue

            matches = bf.knnMatch(des_ref, des_roi, k=2)
            good = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            if len(good) < MIN_CONFIRM_MATCHES:
                tracker = None
                tracking = False
                continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Tracking",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Object Detection & Tracking", frame)
        if cv2.waitKey(1) == 27:
            break
        continue


    # DETECTION MODE
    kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
    if des_frame is None:
        cv2.imshow("Object Detection & Tracking", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    matches = bf.knnMatch(des_ref, des_frame, k=2)

    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) < MIN_DETECT_MATCHES:
        cv2.imshow("Object Detection & Tracking", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None or mask.sum() < MIN_INLIERS:
        cv2.imshow("Object Detection & Tracking", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    corners = np.float32([
        [0, 0],
        [w_ref, 0],
        [w_ref, h_ref],
        [0, h_ref]
    ]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(corners, M)
    x, y, w, h = cv2.boundingRect(np.int32(projected))

    bbox = sanitize_bbox(x, y, w, h, frame_w, frame_h)
    if bbox is None:
        cv2.imshow("Object Detection & Tracking", frame)
        continue

    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)
    tracking = True

    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(
        frame,
        "Detected",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.imshow("Object Detection & Tracking", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
