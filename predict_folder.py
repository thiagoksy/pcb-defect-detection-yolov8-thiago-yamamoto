# Author: Thiago Yamamoto
# Note: 
#   * Loads trained YOLOv8 weights to run batch inference on unseen test images.
#   * save=True: Automatically exports and saves results with bounding boxes into the runs/detect directory.
#   * conf=0.5: Sets the minimum confidence threshold to filter out low-certainty predictions.

from ultralytics import YOLO

if __name__ == "__main__":
    # Load your trained model weights (replace with your actual path)
    model = YOLO(r"the path where your neural link generate")

    # Run batch inference on the test dataset folder
    results = model.predict(
        source=r"path.....\test\images", 
        save=True,       # Save the images with bounding boxes
        conf=0.5         # Minimum confidence threshold
    )

    print("Test completed successfully!")
