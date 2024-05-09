import os


def get_voices_yaml(directory):
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    print(entry.path)
    except FileNotFoundError:
        print("Directory not found")
    except PermissionError:
        print("Permission denied")
    except Exception as e:
        print(f"An error occurred: {e}")


# Specify the path to the directory
directory_path = 'GPT_SoVITS/configs/voices'

# Call the function
get_voices_yaml(directory_path)
