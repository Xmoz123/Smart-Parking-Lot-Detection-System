import os
import shutil

# Define directories
full_image_root = 'FULL_IMAGE_1000x750'
output_labels_root = 'output_yolo_labels'
temp_upload_dir = 'CNRPark_S1_Upload_Temp'

# Clean up existing temp directory if it exists
if os.path.exists(temp_upload_dir):
    print(f"Cleaning existing temp directory: {temp_upload_dir}")
    shutil.rmtree(temp_upload_dir)
os.makedirs(temp_upload_dir, exist_ok=True)

# Counters for summary
total_images_found_in_source = 0
copied_images = 0
copied_actual_labels = 0
created_dummy_labels = 0
labels_not_found_for_copy = 0

print(f"Starting to process images from: {os.path.abspath(full_image_root)}")
print(f"Looking for corresponding labels in: {os.path.abspath(output_labels_root)}")
print(f"Copying to: {os.path.abspath(temp_upload_dir)}")

# Walk through all image files
for root, _, files in os.walk(full_image_root):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            total_images_found_in_source += 1
            image_path_source = os.path.join(root, filename)
            base_filename_no_ext = os.path.splitext(filename)[0]
            original_label_path = os.path.join(output_labels_root, base_filename_no_ext + '.txt')
            image_path_dest = os.path.join(temp_upload_dir, filename)
            label_path_dest = os.path.join(temp_upload_dir, base_filename_no_ext + '.txt')

            # Copy image
            try:
                shutil.copy2(image_path_source, image_path_dest)
                copied_images += 1
            except Exception as e:
                print(f"Error copying image {image_path_source} to {image_path_dest}: {e}")
                continue

            # Copy label if it exists; else create empty dummy label
            if os.path.exists(original_label_path):
                try:
                    shutil.copy2(original_label_path, label_path_dest)
                    copied_actual_labels += 1
                except Exception as e:
                    print(f"Error copying label {original_label_path} to {label_path_dest}: {e}")
                    with open(label_path_dest, 'w') as f_dummy:
                        pass
                    created_dummy_labels += 1
                    labels_not_found_for_copy += 1
            else:
                print(f"Warning: Original label file not found at {original_label_path} for image {image_path_source}.")
                with open(label_path_dest, 'w') as f_dummy:
                    pass
                created_dummy_labels += 1
                labels_not_found_for_copy += 1

# Summary
print(f"\n--- Summary of Populating {temp_upload_dir} ---")
print(f"Total image files found in source ('{full_image_root}'): {total_images_found_in_source}")
print(f"Images successfully copied to '{temp_upload_dir}': {copied_images}")
print(f"Actual label files successfully copied: {copied_actual_labels}")
print(f"Empty dummy label files created (due to missing original labels or copy errors): {created_dummy_labels}")
if labels_not_found_for_copy > 0:
    print(f"Warning: {labels_not_found_for_copy} original label files were not found or failed to copy.")
