import os
import shutil

# ---------------------- CONFIG ------------------------
PATCHES_ROOT_DIR = '.'  # Root directory where PATCHES and LABELS folders exist
CNRPARK_LABELS_DIR = os.path.join(PATCHES_ROOT_DIR, 'LABELS')
OUTPUT_STAGE2_DATA_DIR = os.path.join(PATCHES_ROOT_DIR, 'sorted_patches')

# Mapping of CNRPark labels to folder names
LABEL_TO_FOLDER = {
    '0': 'empty',
    '1': 'occupied'
}

# List of label split files to process
SPLIT_FILES_TO_PROCESS = ['train.txt', 'val.txt', 'test.txt']

# ---------------------- MAIN SCRIPT ------------------------
if __name__ == '__main__':
    # Validate required directories
    if not os.path.isdir(PATCHES_ROOT_DIR):
        print(f"Error: Current directory somehow not found: '{os.path.abspath(PATCHES_ROOT_DIR)}'")
        exit()
    if not os.path.isdir(CNRPARK_LABELS_DIR):
        print(f"Error: CNRPark LABELS directory not found: '{os.path.abspath(CNRPARK_LABELS_DIR)}'")
        exit()
    expected_patches_data_folder = os.path.join(PATCHES_ROOT_DIR, 'PATCHES')
    if not os.path.isdir(expected_patches_data_folder):
        print(f"Error: The 'PATCHES' data subfolder not found at: '{os.path.abspath(expected_patches_data_folder)}'")
        exit()

    # Print directory structure info
    print(f"Using PATCHES_ROOT_DIR: {os.path.abspath(PATCHES_ROOT_DIR)}")
    print(f"Expecting PATCHES: {os.path.abspath(expected_patches_data_folder)}")
    print(f"Using LABELS: {os.path.abspath(CNRPARK_LABELS_DIR)}")
    print(f"Output will be in: {os.path.abspath(OUTPUT_STAGE2_DATA_DIR)}")

    # Counters for summary
    overall_processed_count = 0
    overall_copied_count = 0
    overall_source_not_found_count = 0

    # Process each split file
    for split_filename in SPLIT_FILES_TO_PROCESS:
        label_file_path = os.path.join(CNRPARK_LABELS_DIR, split_filename)
        split_name = os.path.splitext(split_filename)[0]

        if not os.path.exists(label_file_path):
            print(f"\nWarning: Label file not found: '{label_file_path}'. Skipping this split.")
            continue

        print(f"\nProcessing: {label_file_path} (split: {split_name})")
        current_split_copied_count = 0
        current_split_source_not_found = 0

        # Read and process each line of label file
        with open(label_file_path, 'r') as f_in:
            for line_num, line in enumerate(f_in):
                overall_processed_count += 1
                try:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        print(f"  L{line_num+1} Warning: Malformed line. Skipping: '{line.strip()}'")
                        continue

                    relative_patch_path = parts[0]
                    cnrpark_label = parts[1]

                    # Skip unknown labels
                    if cnrpark_label not in LABEL_TO_FOLDER:
                        print(f"  L{line_num+1} Warning: Unknown label '{cnrpark_label}'. Skipping.")
                        continue

                    target_folder_name = LABEL_TO_FOLDER[cnrpark_label]
                    source_patch_full_path = os.path.join(PATCHES_ROOT_DIR, 'PATCHES', relative_patch_path)

                    # Check if source exists
                    if not os.path.exists(source_patch_full_path):
                        print(f"  L{line_num+1} Error: Source patch not found: '{source_patch_full_path}'")
                        current_split_source_not_found += 1
                        overall_source_not_found_count += 1
                        if overall_source_not_found_count == 6:
                            print("    DEBUG: Further 'not found' errors will be suppressed.")
                        continue

                    # Create output directory
                    target_split_dir = os.path.join(OUTPUT_STAGE2_DATA_DIR, split_name, target_folder_name)
                    os.makedirs(target_split_dir, exist_ok=True)

                    # Copy file
                    patch_filename = os.path.basename(relative_patch_path)
                    target_patch_full_path = os.path.join(target_split_dir, patch_filename)
                    shutil.copy2(source_patch_full_path, target_patch_full_path)
                    current_split_copied_count += 1
                    overall_copied_count += 1

                    if current_split_copied_count > 0 and current_split_copied_count % 500 == 0:
                        print(f"    Copied {current_split_copied_count} patches for '{split_name}'...")

                except Exception as e:
                    print(f"  L{line_num+1} Error processing line: {e}")

        # Split summary
        print(f"Finished '{split_filename}' - Copied: {current_split_copied_count}, Not Found: {current_split_source_not_found}")

    # Final summary
    print(f"\n--- Summary ---")
    print(f"Total lines processed: {overall_processed_count}")
    print(f"Total copied: {overall_copied_count}")
    print(f"Total not found: {overall_source_not_found_count}")
    print(f"Output dir: {os.path.abspath(OUTPUT_STAGE2_DATA_DIR)}")
