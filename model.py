import sys  # Import the sys module
import os  # Import the os module
import cv2
import torch
import numpy as np
from ultralytics import YOLO

sys.path.append(os.path.join(os.getcwd(), 'sort'))  # Add sort folder to path

from sort import Sort  # Import SORT tracker

# Load YOLOv8 Model (Pretrained on COCO dataset)
model = YOLO("yolov8n.pt")

# Initialize SORT Tracker
tracker = Sort()

# Object categories from COCO dataset
COCO_CATEGORIES = model.names  # Load class names
OBJECT_CLASSES = ["bottle", "cup", "cell phone", "book", "remote"]  # Objects to track

# Open video file
video_path = "testVdo.gif"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Video Writer setup
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

prev_objects = {}  # Store previous frame objects

# Process Video Frame by Frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Run YOLOv8 detection
    detections = []  # List to store detections

    current_objects = {}  # Store detected objects in the current frame
    human_boxes = []  # List to store human bounding boxes
    picked_up_objects = []  # Store objects that were picked up

    # Process YOLO Detections
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box.tolist()
            class_name = COCO_CATEGORIES[int(class_id)]

            if score > 0.5:
                # Default color for bounding box (Green for person, Blue for others)
                color = (0, 255, 0) if class_name == "person" else (255, 0, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{class_name} {score:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if class_name == "person":
                    human_boxes.append([x1, y1, x2, y2])
                elif class_name in OBJECT_CLASSES:
                    detections.append([x1, y1, x2, y2, score])
                    current_objects[class_name] = (x1, y1, x2, y2)

    # Track Objects Using SORT
    tracked_objects = tracker.update(np.array(detections))

    # Check if any object was picked up
    for obj_name, (x1, y1, x2, y2) in prev_objects.items():
        if obj_name not in current_objects:
            for h_x1, h_y1, h_x2, h_y2 in human_boxes:
                # If object overlaps with a human in this frame, assume it's picked up
                if (x1 > h_x1 and x2 < h_x2 and y1 > h_y1 and y2 < h_y2):
                    cv2.putText(frame, f"Picked Up: {obj_name}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print(f"[INFO] {obj_name} was picked up!")

                    # Change the bounding box of the picked object to red
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red color
                    picked_up_objects.append(obj_name)

    # Update previous objects
    prev_objects = current_objects.copy()

    # Write frame to output video
    out.write(frame)
    cv2.imshow("YOLOv8 Object Pickup Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()


