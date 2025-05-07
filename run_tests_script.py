import subprocess
import os
import glob

def run_command(yaml_file):
    """Runs the command with the given YAML file, printing its contents."""

    # Print the config file contents
    try:
        with open(yaml_file, 'r') as f:
            print(f"Running with config file: {yaml_file}")
    except FileNotFoundError:
        print(f"Error: Config file {yaml_file} not found.")
        return  # Skip this file if it's not found

    command = ["python", "mmlatch/run.py", "--config", yaml_file]
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
    directories = ["mmlatch/configs/mask_tests", "mmlatch/configs/noise_tests", "mmlatch/configs/augment_tests"]
    
    for directory in directories:
        yaml_files = find_yaml_files(directory)
        for yaml_file in yaml_files:
            run_command(yaml_file)

    print("Finished processing all YAML files.")


if __name__ == "__main__":
    main()
