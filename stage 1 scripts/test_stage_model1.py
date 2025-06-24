import cv2
from ultralytics import YOLO
import os
import csv  

# Define paths and parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = 'best.pt'
MODEL_PATH = os.path.join(script_dir, MODEL_FILENAME)
IMAGE_DIR = os.path.join(script_dir, 'test_images')
OUTPUT_DIR = os.path.join(script_dir, 'test_results')
CSV_OUTPUT_DIR = os.path.join(script_dir, 'csv_output')
CSV_OUTPUT_PATH = os.path.join(CSV_OUTPUT_DIR, 'stage1_detections.csv')
CONFIDENCE_THRESHOLD = 0.25

# Ensure necessary directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

def run_inference_on_images(model, image_directory, output_directory, csv_path, conf_threshold):
    """
    Runs inference on images, displays results with annotated legend and bounding boxes,
    saves output images and writes detections to CSV.
    """
    if not os.listdir(image_directory):
        print(f"No images found in '{image_directory}'. Please add some test images.")
        return

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    all_detections_for_csv = []

    # Define legend text properties
    legend_text = "PS: Parking_Slot"
    legend_font_scale = 0.7
    legend_thickness = 2
    legend_text_color = (0, 0, 0)
    legend_bg_color = (220, 220, 220)
    legend_padding = 10

    for filename in os.listdir(image_directory):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(image_directory, filename)
            print(f"Processing image: {image_path}")

            results = model.predict(source=image_path, conf=conf_threshold, save=False)

            for result in results:
                img_to_draw_on = result.orig_img.copy()
                boxes = result.boxes
                custom_class_name = "PS"

                # Draw legend box
                (legend_w, legend_h), _ = cv2.getTextSize(legend_text, cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, legend_thickness)
                cv2.rectangle(img_to_draw_on, 
                              (legend_padding // 2, legend_padding // 2),
                              (legend_padding // 2 + legend_w + legend_padding, legend_padding // 2 + legend_h + legend_padding),
                              legend_bg_color, -1)
                cv2.putText(img_to_draw_on, legend_text,
                            (legend_padding, legend_padding + legend_h),
                            cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, legend_text_color, legend_thickness)

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    label = f'{custom_class_name} {conf:.2f}'
                    box_color = (255, 0, 0)
                    text_color_on_box = (255, 255, 255)
                    font_scale = 0.6
                    thickness = 2

                    # Draw bounding box
                    cv2.rectangle(img_to_draw_on, (x1, y1), (x2, y2), box_color, thickness)

                    # Determine label position
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness -1 if thickness > 1 else 1)
                    text_y_pos = y1 - text_height - baseline - 5
                    if text_y_pos < (legend_padding // 2 + legend_h + legend_padding + 5):
                        text_y_pos = y1 + text_height + 5
                        if text_y_pos + text_height > img_to_draw_on.shape[0]:
                            text_y_pos = y1 - baseline - 5

                    # Draw label background + text
                    cv2.rectangle(img_to_draw_on, (x1, text_y_pos - text_height - baseline), (x1 + text_width, text_y_pos + baseline), box_color, -1)
                    cv2.putText(img_to_draw_on, label, (x1 + 2, text_y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color_on_box, thickness -1 if thickness > 1 else 1)

                    # Record detection for CSV
                    all_detections_for_csv.append({
                        'image_filename': filename,
                        'class_id': int(box.cls[0]),
                        'class_name': custom_class_name,
                        'confidence': f"{conf:.4f}",
                        'x_min': x1,
                        'y_min': y1,
                        'x_max': x2,
                        'y_max': y2,
                        'width': x2 - x1,
                        'height': y2 - y1
                    })

                # Show and save result
                cv2.imshow(f'Detections - {filename}', img_to_draw_on)
                print(f"  Found {len(result.boxes)} parking slots.")
                print("  Press any key to continue to the next image (or 'q' to quit)...")
                key = cv2.waitKey(0)
                output_path = os.path.join(output_directory, f"detected_{filename}")
                cv2.imwrite(output_path, img_to_draw_on)
                print(f"  Result saved to: {output_path}")

                if key == ord('q'):
                    print("Quitting...")
                    cv2.destroyAllWindows()
                    return

    # Write detections to CSV
    if all_detections_for_csv:
        fieldnames = ['image_filename', 'class_id', 'class_name', 'confidence',
                      'x_min', 'y_min', 'x_max', 'y_max', 'width', 'height']
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_detections_for_csv)
        print(f"\nAll Stage 1 detections saved to CSV: {csv_path}")
    else:
        print("\nNo detections to save to CSV.")

    cv2.destroyAllWindows()
    print("Finished processing all images.")

if __name__ == '__main__':
    try:
        print(f"Loading model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure '{MODEL_FILENAME}' is in the directory: '{os.path.abspath(script_dir)}'")
        exit()

    run_inference_on_images(model, IMAGE_DIR, OUTPUT_DIR, CSV_OUTPUT_PATH, CONFIDENCE_THRESHOLD)
