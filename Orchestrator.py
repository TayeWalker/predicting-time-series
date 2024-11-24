import os
import subprocess

# Base paths for the folders
base_path = os.path.dirname(os.path.abspath(__file__))
multivariate_path = os.path.join(base_path, "Multivariate_MOIRAI")
univariate_path = os.path.join(base_path, "Univariate_MOIRAI")

uni_predictions = os.path.join(base_path, "Univariate_MOIRAI/forecast.npy")
multi_predictions = os.path.join(base_path, "Multivariate_MOIRAI/forecast.npy")
ground_truth = os.path.join(base_path, "Multivariate_MOIRAI/truth.npy")

# Unified configuration file
config_path = os.path.join(base_path, "config.json")

# Docker images
docker_images = {
    "multivariate": "multivariate_moirai_image",
    "univariate": "univariate_moirai_image",
}

def build_docker(container_type):
    """
    Build the Docker image for the specified container type.
    """
    if container_type not in ["multivariate", "univariate"]:
        raise ValueError(f"Invalid container type: {container_type}")
    
    folder_path = multivariate_path if container_type == "multivariate" else univariate_path
    docker_image = docker_images[container_type]
    
    print(f"Building Docker image for {container_type}...")
    
    try:
        subprocess.run(
            ["docker", "build", "-t", docker_image, folder_path],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error building {container_type} image: {e}")
        raise

def run_docker(container_type):
    """
    Run a Docker container for the specified container type.
    """
    if container_type not in ["multivariate", "univariate"]:
        raise ValueError(f"Invalid container type: {container_type}")
    
    folder_path = multivariate_path if container_type == "multivariate" else univariate_path
    docker_image = docker_images[container_type]
    
    print(f"Running {container_type} predictions...")
    print(f"Folder: {folder_path}")
    print(f"Config file: {config_path}")
    
    try:
        # Run the Docker container
        subprocess.run(
            [
                "docker", "run",
                "-v", f"{config_path}:/app/config.json",  # Mount the unified config.json
                "-v", f"{folder_path}:/app",  # Mount the folder
                docker_image
            ],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running {container_type} container: {e}")
        raise

def run_evaluation():
    print("Running evaluation...")
    subprocess.run(
        [
            "python", os.path.join(base_path, "comparison/evaluation.py"),
            "--uni", uni_predictions,
            "--multi", multi_predictions,
            "--truth", ground_truth
        ],
        check=True
    )

if __name__ == "__main__":
    # Build and run multivariate container
    build_docker("multivariate")
    run_docker("multivariate")
    
    # Build and run univariate container
    build_docker("univariate")
    run_docker("univariate")

    run_evaluation()
