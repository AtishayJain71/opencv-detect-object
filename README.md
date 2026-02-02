# Object Detection and Tracking using OpenCV

This project implements a classical computer vision system to detect and track a known reference object (for example, a Frooti tetra pack) in a video file or a live camera feed using OpenCV.  
The system first detects the object using feature-based matching and then tracks it in real time using a bounding box.

The solution does not use any deep learning or model training and relies entirely on traditional OpenCV techniques.

---

## Features

- Detects a given reference object in a video or live camera feed
- Continuously tracks the object in real time
- Draws a bounding box around the detected/tracked object
- Automatically re-detects the object if tracking fails
- Supports both prerecorded videos and live webcam input
- Implemented using only OpenCV and NumPy

---

## Requirements

- Python 3.8+
- OpenCV (contrib modules)
- NumPy

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Project Structure
```
.
├── data/
│   └── reference.jpg              # Reference image of the object
├── demo/
│   └── demo.mp4              # Sample input video
├── src/
│   └── detect_and_track.py   # Main source code
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### 1. Run on a Video File

In `detect_and_track.py`, set:

```python
USE_CAMERA = False
VIDEO_PATH = "demo/demo.mp4"
```

Then run:

```bash
python src/detect_and_track.py
```

### 2. Run on Live Camera Feed

In `detect_and_track.py`, set:

```python
USE_CAMERA = True
CAMERA_INDEX = 0
```

Then run:

```bash
python src/detect_and_track.py
```

Press `ESC` to exit the application.

---

## Key Design Decisions

### 1. Feature-Based Object Detection (SIFT)

SIFT (Scale-Invariant Feature Transform) is used to extract robust local features from the reference image and video frames.

SIFT provides invariance to scale and rotation, making it suitable for detecting the same object at different distances and orientations.

Feature matching combined with homography estimation (RANSAC) is used to confirm object presence and compute the object’s bounding box.

### 2. Tracking After Detection

Once the object is detected, a CSRT tracker is initialized.

Tracking is significantly faster than running detection on every frame.

This allows smooth real-time tracking and helps handle temporary motion blur.

### 3. Periodic Validation

During tracking, feature matching is periodically performed inside the tracked region.

If validation fails, the tracker is discarded and detection is re-triggered.

This prevents tracker drift and ensures the bounding box does not persist when the object leaves the frame.

### 4. Performance Optimizations

- Frames are resized before processing to improve speed.
- The number of SIFT features is reduced to balance accuracy and performance.
- Grayscale conversion is performed once per frame to avoid redundant computation.

---

## Known Limitations

This system uses classical feature-based methods and does not understand objects semantically.

Detection accuracy on live camera feeds is sensitive to:

- Motion blur
- Lighting changes
- Background clutter

If the reference image contains background regions, similar background textures in the live camera feed may cause false matches.

The approach is not optimized for long-duration CCTV or large-scale surveillance scenarios.

These limitations are inherent to non-learning, feature-based computer vision approaches.

---

## What I Would Improve With More Time

- Crop or mask the reference image to remove background features
- Use multiple reference images from different viewpoints
- Improve scale estimation during tracking
- Further optimize the pipeline for real-time live camera feeds
- Explore learning-based object detectors

---

## Demo

A short demo video/GIF is included in the repository showing the output of the program, including:

- Initial object detection
- Continuous tracking with a bounding box
- Recovery after temporary tracking failure

The demo demonstrates the system running on a sample video or live camera feed, as required by the assignment.