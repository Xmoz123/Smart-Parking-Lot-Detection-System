from ultralytics import YOLO
import os
import yaml

# Set paths relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, 'best.pt')
DATA_YAML_PATH = os.path.join(script_dir, 'data.yaml')
SPLIT_TO_EVALUATE = 'test'

if __name__ == '__main__':
    # Check required files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Error: data.yaml file not found at {DATA_YAML_PATH}")
        print("Ensure this YAML file defines the path to your test set images and labels.")
        exit()

    # Load YOLO model
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")

    # Run validation/evaluation on specified split
    print(f"\nStarting evaluation on '{SPLIT_TO_EVALUATE}' split using {DATA_YAML_PATH}...")
    metrics = model.val(
        data=DATA_YAML_PATH,
        split=SPLIT_TO_EVALUATE,
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.45,
    )

    # Print key metrics
    print("\n--- Evaluation Metrics ---")
    if metrics and metrics.box:
        print(f"  Precision (P):            {metrics.box.mp:.4f}")
        print(f"  Recall (R):               {metrics.box.mr:.4f}")
        print(f"  mAP50 (mean Avg Prec @ IoU=0.50): {metrics.box.map50:.4f}")
        print(f"  mAP50-95 (mean Avg Prec @ IoU=0.50:0.95): {metrics.box.map:.4f}")
        # Calculate F1 if precision + recall > 0
        if (metrics.box.mp + metrics.box.mr) > 0:
            f1_score = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr)
            print(f"  F1 Score (calculated from P & R): {f1_score:.4f}")
    else:
        print("Could not retrieve detailed box metrics.")
        print("Full metrics object:", metrics)

    print("\nEvaluation complete. Detailed results and plots saved in a 'runs/detect/valX' folder.")
