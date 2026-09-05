# Author: Thiago Yamamoto
# Note: 
#   * Runs real-time inference using a connected webcam feed (source=0).
#   * show=True: Opens an interactive OpenCV window displaying live detections.
#   * conf=0.5: Sets the minimum confidence threshold to filter out weak predictions in real-time.
#   * Press 'q' while focusing on the video window to stop execution.

from ultralytics import YOLO

if __name__ == "__main__":
    # Load your trained YOLOv8 model weights
    # Replace the path below with your local path or relative path to best.pt
    model = YOLO(r"the path where your neural link generatet")

    # Execute real-time inference using webcam (device index 0)
    print("Starting webcam feed... Press 'q' on the video window to exit.")
    results = model.predict(
        source=0,     # Index 0 selects the default system webcam
        show=True,    # Displays the live video window with bounding boxes
        conf=0.5      # Minimum confidence threshold (50%)
    )
