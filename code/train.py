from ultralytics import YOLO
import os

def main():
    model_path = os.path.join("..", "models", "cricket_ball_final", "weights", "best.pt")

    print(f"Loading weights from: {model_path}")
    model = YOLO(model_path) 

    # Path to yaml
    yaml_path = os.path.abspath(os.path.join("..", "dataset", "data.yaml"))

    print("Resuming Training for 20 more epochs...")
    
    model.train(
        data=yaml_path,
        epochs=20,          # Adding 20 more epochs
        imgsz=512,          
        batch=16,
        project=os.path.abspath(os.path.join("..", "models")),
        name="cricket_ball_refined", # New folder for refined model
        device="cpu",
        plots=True
    )

if __name__ == '__main__':
    main()