import subprocess
import os
import glob

def find_output_dirs(directory):
    """Finds all output directories in the given directory and returns only their names."""
    # Get the full paths of all directories
    dir_paths = glob.glob(os.path.join(directory, "*"))
    
    # Extract only the directory names
    dir_names = [os.path.basename(dir_path) for dir_path in dir_paths if os.path.isdir(dir_path)]
    
    # Include special case: "no_mmlatch_test"
    dir_names.append("no_mmlatch")

    return dir_names

def run_command(yaml_file):
    """Runs the command with the given YAML file, printing its contents."""

    exp_name = yaml_file.split('/')[-1][:-5]
    output_dirs = find_output_dirs("/content/drive/MyDrive/our_results")

    # Print the config file contents
    try:
        with open(yaml_file, 'r') as f:
            print(f"Running with config file: {yaml_file}")
    except FileNotFoundError:
        print(f"Error: Config file {yaml_file} not found.")
        return  # Skip this file if it's not found
    
    if exp_name in output_dirs:
        print(f"{exp_name} already completed")
        return
    
    command = ["python", "mmlatch/run_mosei.py", "--config", yaml_file]
    print(f"Executing: {' '.join(command)}")  # Print the command

    try:
        # Use os.path.abspath to ensure correct paths
        command = [os.path.abspath(arg) if os.path.exists(arg) else arg for arg in command]

        subprocess.run(command, check=True)  # Execute and check for errors
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)  # Print stdout
        print("--------------------")  # Separator
    except subprocess.CalledProcessError as e:
        print(f"Error running command for {yaml_file}: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)


def find_yaml_files(directory):
    """Finds all YAML files in the given directory."""
    return glob.glob(os.path.join(directory, "*.yaml"))


def main():
    """Main function to process all YAML files."""
    directories = ["mmlatch/configs/mask_tests", "mmlatch/configs/augment_tests","mmlatch/configs/noise_tests"]
    run_command("mmlatch/configs/config_MOSEI.yaml")


    for directory in directories:
        yaml_files = find_yaml_files(directory)
        for yaml_file in yaml_files:
            run_command(yaml_file)

    print("Finished processing all YAML files.")


if __name__ == "__main__":
    main()
