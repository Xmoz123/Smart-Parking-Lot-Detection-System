import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import csv

# ---------------------- CONFIG ------------------------
STAGE1_MODEL_PATH = 'best.pt'
STAGE2_MODEL_PATH = 'stage2_occupancy_classifier_best.h5'
INPUT_TEST_IMAGES_DIR = 'test_images'
OUTPUT_VISUALIZATION_DIR = 'test_results_creative'
OUTPUT_CSV_DIR = 'csv_output'

STAGE2_IMG_HEIGHT = 96
STAGE2_IMG_WIDTH = 96

# Create output directories if not exist
os.makedirs(OUTPUT_VISUALIZATION_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# ---------------------- MODEL LOADING ------------------------
print(f"Loading Stage 1 YOLOv8 model from: {STAGE1_MODEL_PATH}")
try:
    stage1_model = YOLO(STAGE1_MODEL_PATH)
    print("Stage 1 YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading Stage 1 model: {e}")
    exit()

print(f"Loading Stage 2 CNN Occupancy model from: {STAGE2_MODEL_PATH}")
try:
    stage2_model = load_model(STAGE2_MODEL_PATH)
    print("Stage 2 CNN Occupancy model loaded successfully.")
except Exception as e:
    print(f"Error loading Stage 2 model: {e}")
    exit()

# ---------------------- MAIN INFERENCE FUNCTION ------------------------
def predict_parking_occupancy_creative(image_path_or_cv2_image, stage1_conf=0.3, stage2_occupied_threshold=0.7):
    # Handle input type (path or cv2 image)
    if isinstance(image_path_or_cv2_image, str):
        original_image = cv2.imread(image_path_or_cv2_image)
        if original_image is None:
            print(f"Error: Could not read image from {image_path_or_cv2_image}")
            return None, None, 0, 0
    else:
        original_image = image_path_or_cv2_image.copy()

    # Run YOLOv8 detection
    stage1_results = stage1_model.predict(original_image, conf=stage1_conf, iou=0.5, verbose=False)

    detected_slots_info = []
    output_visualization_image = original_image.copy()
    occupied_count_viz = 0
    empty_count_viz = 0

    # Process detections if any
    if stage1_results and stage1_results[0].boxes and len(stage1_results[0].boxes) > 0:
        boxes = stage1_results[0].boxes.xyxy.cpu().numpy()
        confidences_s1 = stage1_results[0].boxes.conf.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Crop the detected region
            h_img, w_img = original_image.shape[:2]
            x1_crop, y1_crop = max(0, x1), max(0, y1)
            x2_crop, y2_crop = min(w_img, x2), min(h_img, y2)

            if x1_crop >= x2_crop or y1_crop >= y2_crop:
                continue
            slot_crop = original_image[y1_crop:y2_crop, x1_crop:x2_crop]
            if slot_crop.size == 0:
                continue

            # Prepare crop for stage 2 CNN classifier
            img_resized_for_stage2 = cv2.resize(slot_crop, (STAGE2_IMG_WIDTH, STAGE2_IMG_HEIGHT))
            img_array_for_stage2 = img_to_array(img_resized_for_stage2) / 255.0
            img_batch_for_stage2 = np.expand_dims(img_array_for_stage2, axis=0)

            # Run stage 2 prediction
            prediction_s2_raw = stage2_model.predict(img_batch_for_stage2, verbose=0)
            prediction_s2 = prediction_s2_raw[0][0]

            # Determine occupancy
            occupancy_status = "occupied" if prediction_s2 > stage2_occupied_threshold else "empty"
            detected_slots_info.append({'box': [x1, y1, x2, y2], 'status': occupancy_status})

            # Draw marker + optional box
            marker_radius = 8
            line_thickness_for_box = 1

            if occupancy_status == "occupied":
                color = (0, 0, 255)
                occupied_count_viz += 1
                if line_thickness_for_box > 0:
                    cv2.rectangle(output_visualization_image, (x1, y1), (x2, y2), (50, 50, 150), line_thickness_for_box)
                cv2.circle(output_visualization_image, (center_x, center_y), marker_radius, color, -1)
            else:
                color = (0, 255, 0)
                empty_count_viz += 1
                if line_thickness_for_box > 0:
                    cv2.rectangle(output_visualization_image, (x1, y1), (x2, y2), (50, 150, 50), line_thickness_for_box)
                cv2.circle(output_visualization_image, (center_x, center_y), marker_radius, color, -1)
    else:
        print("  Stage 1: No slots detected for this image.")

    # Add summary text on image
    total_detected_viz = occupied_count_viz + empty_count_viz
    summary_text = f"Detected: {total_detected_viz} | Occupied: {occupied_count_viz} | Empty: {empty_count_viz}"
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    text_thickness = 2
    text_color = (255, 255, 255)
    background_color = (0, 0, 0)

    (text_width, text_height), baseline = cv2.getTextSize(summary_text, font_face, font_scale, text_thickness)
    text_x = 10
    text_y = 10 + text_height

    cv2.rectangle(output_visualization_image, (text_x - 5, text_y - text_height - baseline + 5),
                  (text_x + text_width + 5, text_y + baseline - 5), background_color, -1)
    cv2.putText(output_visualization_image, summary_text, (text_x, text_y),
                font_face, font_scale, text_color, text_thickness, cv2.LINE_AA)

    return output_visualization_image, detected_slots_info, occupied_count_viz, empty_count_viz

# ---------------------- MAIN SCRIPT ------------------------
if __name__ == '__main__':
    if not os.path.isdir(INPUT_TEST_IMAGES_DIR) or not os.listdir(INPUT_TEST_IMAGES_DIR):
        print(f"Error: Test images directory '{INPUT_TEST_IMAGES_DIR}' not found or is empty.")
    else:
        print(f"Processing images from: {INPUT_TEST_IMAGES_DIR}")
        all_images_summary_for_csv = []

        for image_filename_with_ext in os.listdir(INPUT_TEST_IMAGES_DIR):
            if image_filename_with_ext.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_image_full_path = os.path.join(INPUT_TEST_IMAGES_DIR, image_filename_with_ext)
                print(f"\n--- Processing: {test_image_full_path} ---")

                base_name = os.path.splitext(image_filename_with_ext)[0]
                output_viz_filename = f"{base_name}_creative_occupancy.jpg"

                result_image, slots_details, occupied_final, empty_final = predict_parking_occupancy_creative(
                    test_image_full_path,
                    stage1_conf=0.2,
                    stage2_occupied_threshold=0.9
                )

                if result_image is not None:
                    viz_save_path = os.path.join(OUTPUT_VISUALIZATION_DIR, output_viz_filename)
                    cv2.imwrite(viz_save_path, result_image)
                    print(f"  Output visualization saved to: {viz_save_path}")

                    all_images_summary_for_csv.append({
                        'Image Name': image_filename_with_ext,
                        'Total Detected Slots': occupied_final + empty_final,
                        'Occupied Slots': occupied_final,
                        'Available Slots': empty_final
                    })
            else:
                print(f"Skipping non-image file: {image_filename_with_ext}")

        if all_images_summary_for_csv:
            overall_csv_path = os.path.join(OUTPUT_CSV_DIR, "all_images_parking_summary_creative.csv")
            fieldnames = ['Image Name', 'Total Detected Slots', 'Occupied Slots', 'Available Slots']
            with open(overall_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_images_summary_for_csv)
            print(f"\nOverall summary saved to: {overall_csv_path}")
        else:
            print("\nNo images were processed to create an overall summary.")

        print("\nAll specified test images processed.")
