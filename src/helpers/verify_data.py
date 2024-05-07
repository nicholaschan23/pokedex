import os


def remove_empty_dirs(directory: str):
    # Function to recursively find empty directories
    def find_empty_directories(directory):
        empty_directories = []
        for root, dirs, files in os.walk(directory, topdown=False):
            if not dirs and not files:
                empty_directories.append(root)
        return empty_directories

    # Find empty directories
    empty_dirs = find_empty_directories(directory)

    # Remove empty directories
    for empty_dir in empty_dirs:
        os.rmdir(empty_dir)
        print(f"Removed empty directory: {empty_dir}")


def remove_invalid_imgs(directory: str):
    # Supported extensions
    IMG_EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    )

    # Function to check if a file has a valid extension
    def is_valid_extension(filename):
        _, extension = os.path.splitext(filename)
        return extension.lower() in IMG_EXTENSIONS

    # Function to recursively search for files in nested directories
    def search_files(directory):
        invalid_files = []
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if not is_valid_extension(filename):
                    invalid_file_path = os.path.join(root, filename)
                    invalid_files.append(invalid_file_path)
                    os.remove(invalid_file_path)  # Delete the invalid file
        return invalid_files

    # Find invalid files
    invalid_files = search_files(directory)

    # Print invalid files
    if invalid_files:
        print("Deleted invalid files:")
        for file in invalid_files:
            print(file)
    else:
        print("All files have valid extensions.")
