import os
import glob
import shutil

def check_and_move_malformed_yolo_labels(dataset_root_dir, malformed_output_root_dir, splits_to_check=None):
    """
    Checks YOLO label files for malformed lines (not exactly 5 columns).
    Moves malformed label files and their corresponding images to a separate directory.
    """
    # Auto-detect train/valid/test splits if not provided
    if splits_to_check is None:
        splits_to_check = []
        for potential_split in ['train', 'valid', 'test']:
            if os.path.isdir(os.path.join(dataset_root_dir, potential_split)):
                splits_to_check.append(potential_split)

    print(f"Checking splits: {splits_to_check}")
    print(f"Malformed files and their images will be moved to: {os.path.abspath(malformed_output_root_dir)}")

    malformed_files_summary = {}
    total_labels_checked = 0
    total_lines_checked = 0
    total_malformed_lines_overall = 0
    total_files_moved = 0

    for split in splits_to_check:
        source_label_dir = os.path.join(dataset_root_dir, split, 'labels')
        source_image_dir = os.path.join(dataset_root_dir, split, 'images')
        dest_malformed_label_dir = os.path.join(malformed_output_root_dir, split, 'labels')
        dest_malformed_image_dir = os.path.join(malformed_output_root_dir, split, 'images')

        if not os.path.isdir(source_label_dir):
            print(f"Warning: Label directory not found for split '{split}': {source_label_dir}")
            continue
        if not os.path.isdir(source_image_dir):
            print(f"Warning: Image directory not found for split '{split}': {source_image_dir}")

        print(f"\n--- Checking split: {split} ---")
        malformed_files_in_split_details = []
        label_files = glob.glob(os.path.join(source_label_dir, '*.txt'))
        if not label_files:
            print(f"No label files (.txt) found in {source_label_dir}")
            continue

        for label_path in label_files:
            total_labels_checked += 1
            base_label_filename = os.path.basename(label_path)
            image_filename_base = os.path.splitext(base_label_filename)[0]
            current_file_has_malformed_lines = False
            malformed_line_numbers_in_file = []

            try:
                with open(label_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        total_lines_checked += 1
                        parts = line.strip().split()
                        if len(parts) != 5:
                            total_malformed_lines_overall += 1
                            current_file_has_malformed_lines = True
                            malformed_line_numbers_in_file.append(line_num)
            except Exception as e:
                print(f"  ERROR reading or processing file '{label_path}': {e}")
                current_file_has_malformed_lines = True

            if current_file_has_malformed_lines:
                corresponding_image_source_path = None
                image_filename_with_ext = "Unknown"
                for img_ext in ['.jpg', '.jpeg', '.png']:
                    img_path_candidate = os.path.join(source_image_dir, image_filename_base + img_ext)
                    if os.path.exists(img_path_candidate):
                        corresponding_image_source_path = img_path_candidate
                        image_filename_with_ext = os.path.basename(corresponding_image_source_path)
                        break

                malformed_files_in_split_details.append({
                    'label_file_source': label_path,
                    'image_file_source': corresponding_image_source_path,
                    'image_filename': image_filename_with_ext,
                    'label_filename': base_label_filename,
                    'malformed_lines_count': len(malformed_line_numbers_in_file),
                    'first_few_malformed_line_numbers': malformed_line_numbers_in_file[:5]
                })

                os.makedirs(dest_malformed_label_dir, exist_ok=True)
                os.makedirs(dest_malformed_image_dir, exist_ok=True)

                dest_label_path = os.path.join(dest_malformed_label_dir, base_label_filename)
                dest_image_path = None
                if corresponding_image_source_path and os.path.exists(corresponding_image_source_path):
                    dest_image_path = os.path.join(dest_malformed_image_dir, image_filename_with_ext)

                try:
                    shutil.move(label_path, dest_label_path)
                    print(f"  MOVED Label: '{label_path}' -> '{dest_label_path}'")
                    total_files_moved += 1
                    if dest_image_path:
                        shutil.move(corresponding_image_source_path, dest_image_path)
                        print(f"  MOVED Image: '{corresponding_image_source_path}' -> '{dest_image_path}'")
                        total_files_moved += 1
                    elif corresponding_image_source_path:
                        print(f"  Warning: Could not move image for {label_path}, source image not found: {corresponding_image_source_path}")
                except Exception as e:
                    print(f"  ERROR moving file '{label_path}' or its image: {e}")

        if malformed_files_in_split_details:
            print(f"  Moved {len(malformed_files_in_split_details)} label file(s) (and their images if found) from split '{split}'.")
            malformed_files_summary[split] = malformed_files_in_split_details
        else:
            print(f"  No malformed label files found in split '{split}'.")

    print("\n--- Overall Summary ---")
    print(f"Total label files checked: {total_labels_checked}")
    print(f"Total lines checked: {total_lines_checked}")
    print(f"Total malformed lines found: {total_malformed_lines_overall}")
    print(f"Total files moved: {total_files_moved // 2 if total_files_moved > 0 else 0} pairs (approx)")

    if total_malformed_lines_overall > 0:
        print("\nDetails of files with malformed lines:")
        for split, files_details in malformed_files_summary.items():
            print(f"\n  Original Split: {split}")
            for entry in files_details:
                print(f"    Label: {entry['label_file_source']}")
                print(f"    Image: {entry['image_file_source'] if entry['image_file_source'] else 'Not found'}")
                print(f"    Malformed lines: {entry['malformed_lines_count']}")
    else:
        print("All label files are correctly formatted.")

    return malformed_files_summary

if __name__ == '__main__':
    ROBOFLOW_DATASET_ROOT = 'C:\\temp\\pklot dataset for stage 1'
    MALFORMED_OUTPUT_DIR = os.path.join(ROBOFLOW_DATASET_ROOT, 'malformed_files')

    if ROBOFLOW_DATASET_ROOT == 'C:\\temp\\pklot dataset for stage 1' and not os.path.exists(ROBOFLOW_DATASET_ROOT):
        print("Please update 'ROBOFLOW_DATASET_ROOT' in the script with the correct path to your unzipped Roboflow dataset!")
    else:
        print(f"Dataset Root: {os.path.abspath(ROBOFLOW_DATASET_ROOT)}")
        print(f"Malformed files will be moved to: {os.path.abspath(MALFORMED_OUTPUT_DIR)}")
        check_and_move_malformed_yolo_labels(ROBOFLOW_DATASET_ROOT, MALFORMED_OUTPUT_DIR)
