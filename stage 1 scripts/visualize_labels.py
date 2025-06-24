import cv2
import os
import glob

# Helper function to convert YOLO normalized bbox (center_x, center_y, width, height)
# to absolute pixel coordinates (x_min, y_min, x_max, y_max)
def denormalize_yolo_bbox(x_center_norm, y_center_norm, width_norm, height_norm, img_width, img_height):
    abs_x_center = x_center_norm * img_width
    abs_y_center = y_center_norm * img_height
    abs_width = width_norm * img_width
    abs_height = height_norm * img_height

    x_min = int(abs_x_center - (abs_width / 2))
    y_min = int(abs_y_center - (abs_height / 2))
    x_max = int(abs_x_center + (abs_width / 2))
    y_max = int(abs_y_center + (abs_height / 2))

    return x_min, y_min, x_max, y_max

# Main function to visualize YOLO bounding box labels on images
def visualize_yolo_labels(image_dir_to_check, label_root_dir, num_images_to_show=10, class_names=None):
    # Gather image file paths with supported extensions
    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir_to_check, ext)))
    image_files = sorted(image_files)

    # Exit early if no images found
    if not image_files:
        print(f"No images found in {image_dir_to_check} with extensions .jpg, .png, or .jpeg")
        return

    count = 0
    for image_path in image_files:
        # Stop if display limit reached
        if num_images_to_show > 0 and count >= num_images_to_show:
            break

        # Construct corresponding label file path
        base_filename_with_ext = os.path.basename(image_path)
        base_filename_no_ext = os.path.splitext(base_filename_with_ext)[0]
        label_path = os.path.join(label_root_dir, base_filename_no_ext + '.txt')

        # Skip if label file not found
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {image_path} at {label_path}")
            continue

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue

        img_height, img_width, _ = image.shape

        # Read and draw labels
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"  Warning: Malformed line {line_num+1} in {label_path}: '{line.strip()}' (expected 5 parts)")
                    continue

                class_id = int(parts[0])
                x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])

                # Convert normalized bbox to pixel coords
                x_min, y_min, x_max, y_max = denormalize_yolo_bbox(
                    x_center_norm, y_center_norm, width_norm, height_norm,
                    img_width, img_height
                )

                # Draw rectangle
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Prepare label text
                label_text_to_display = str(class_id)
                if class_names and 0 <= class_id < len(class_names):
                    label_text_to_display = f"{class_names[class_id]}({class_id})"

                # Draw label text
                text_y_pos = y_min - 10 if y_min > 20 else y_min + 20
                cv2.putText(image, label_text_to_display, (x_min, text_y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show the image with labels
        cv2.imshow(f'Labeled Image - {base_filename_with_ext}', image)
        key = cv2.waitKey(0)

        # Check if user closed window manually
        if cv2.getWindowProperty(f'Labeled Image - {base_filename_with_ext}', cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed by user. Exiting visualization.")
            break

        cv2.destroyWindow(f'Labeled Image - {base_filename_with_ext}')

        # Quit if 'q' pressed
        if key == ord('q'):
            print("Exiting visualization by 'q' key press.")
            break

        count += 1

    cv2.destroyAllWindows()

    # Summary after completion
    if count == 0 and num_images_to_show != 0:
        print("No images were processed or displayed. Check paths and image extensions.")
    elif num_images_to_show > 0 and count < num_images_to_show:
        print(f"Displayed {count} images. End of available images in '{image_dir_to_check}' or files with matching labels.")
    else:
        print(f"Finished displaying images or reached display limit.")

# Config parameters
IMAGE_DIR_TO_VISUALIZE = 'FULL_IMAGE_1000x750/OVERCAST/2015-11-20/camera1'
LABEL_ROOT_DIRECTORY = 'output_yolo_labels'
NUMBER_OF_IMAGES_TO_DISPLAY = 0  # 0 = show all
CLASS_NAMES = ['parking_slot']

# Entry point
if __name__ == '__main__':
    if not os.path.isdir(IMAGE_DIR_TO_VISUALIZE):
        print(f"Error: Image directory for visualization not found at '{IMAGE_DIR_TO_VISUALIZE}'")
    elif not os.path.isdir(LABEL_ROOT_DIRECTORY):
        print(f"Error: Label root directory not found at '{LABEL_ROOT_DIRECTORY}'")
    else:
        visualize_yolo_labels(IMAGE_DIR_TO_VISUALIZE, LABEL_ROOT_DIRECTORY, NUMBER_OF_IMAGES_TO_DISPLAY, CLASS_NAMES)
