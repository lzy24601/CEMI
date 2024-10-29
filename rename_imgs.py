import os


def rename_images(folder_path):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Filter out only the image files (you can extend this list based on your image formats)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Sort the image files to maintain order
    image_files.sort(key=lambda x: int(x.split('_')[0]), reverse=True)

    # Rename the image files with incremental numbers
    for i, old_name in enumerate(image_files, start=1):
        # Get the file extension
        _, extension = os.path.splitext(old_name)

        # Create the new name with an incremental number and the original extension
        new_name = f"{i:03d}{extension}"

        # Create the full paths for the old and new names
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)

        print(f'Renamed: {old_name} -> {new_name}')


def main():
    # Replace 'your_folder_path' with the path to your folder containing images
    folder_path = r'D:\code\python\EMHD\SR\SR\syn_22_39'

    # Get the list of subdirectories (each subdirectory is considered as a separate batch)
    subdirectories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    # Process each subdirectory
    for subdir in subdirectories:
        subdir_path = os.path.join(folder_path, subdir)
        rename_images(subdir_path)


if __name__ == '__main__':
    main()
