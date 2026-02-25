## This script is only for cleaning the dataset by removing corrupted images and macOS metadata files.
import os
from PIL import Image
import sys

def clean_dataset(dataset_path):
    removed_count = 0
    corrupted_count = 0

    print(f"Scanning dataset at: {dataset_path}\n")

    for root, dirs, files in os.walk(dataset_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for filename in files:
            filepath = os.path.join(root, filename)

            # Remove macOS metadata files
            if filename.startswith('._'):
                print(f"Removing macOS metadata file: {filepath}")
                os.remove(filepath)
                removed_count += 1
                continue

            # Check if it's supposed to be an image file
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                try:
                    # Try to open the image
                    with Image.open(filepath) as img:
                        img.verify()  # Verify it's a valid image

                    # Re-open to ensure it can be loaded (verify() can't be followed by normal operations)
                    with Image.open(filepath) as img:
                        img.load()

                except Exception as e:
                    print(f"Removing corrupted image: {filepath}")
                    print(f"  Error: {str(e)}")
                    os.remove(filepath)
                    corrupted_count += 1

    print(f"\n{'='*60}")
    print(f"Cleanup Summary:")
    print(f"{'='*60}")
    print(f"macOS metadata files removed: {removed_count}")
    print(f"Corrupted images removed: {corrupted_count}")
    print(f"Total files removed: {removed_count + corrupted_count}")
    print(f"{'='*60}")

    return removed_count + corrupted_count

if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else './Jaguar'

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist!")
        sys.exit(1)

    total_removed = clean_dataset(dataset_path)

    if total_removed > 0:
        print(f"\n Dataset cleaned successfully! Please re-run your training.")
    else:
        print(f"\n No corrupted files found. Dataset is clean.")
