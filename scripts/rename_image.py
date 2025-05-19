import os
import sys

def rename_images(folder_path, base_name):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # Get all files in the specified folder
    files = os.listdir(folder_path)
    # Filter for image files with extensions png, jpg, jpeg
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Rename each image file
    for i, file_name in enumerate(image_files, start=1):
        # Get the file extension
        _, ext = os.path.splitext(file_name)
        # Create the new file name
        new_name = f"{base_name}_{i}{ext}"
        # Rename the file
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_name}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rename_image.py <folder_path> <base_name>")
    else:
        rename_images(sys.argv[1], sys.argv[2])