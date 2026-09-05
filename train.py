#  Author: Thiago Yamamoto
#  Note: 
#    * These parameters optimize performance based on specific hardware constraints
#    * Device="cuda": Forces execution exclusively on the GPU, avoiding CPU bottlenecks.
#    * Batch=8 & workers=2: Optimized configuration to prevent out-of-memory (OOM) errors on a 4GB VRAM GPU (GTX 1650) (my user-case)

from ultralytics import YOLO

if __name__ == "__main__":
    # Load a pretrained YOLOv8 nano model as a starting point (transfer learning)
    model = YOLO("yolov8n.pt")

    # Train the model using your custom dataset configuration file
    # Replace 'PCB Defect/data.yaml' with the exact path/folder name created when you extracted your zip
    results = model.train(  # parameters
        data="PCB Defect/data.yaml",
        epochs=15,
        imgsz=416,
        batch=8,
        workers = 2,
        device="cuda",
        name="pcb_defect_model_15_epochs",
    ) 

     print("Training completed successfully!")
