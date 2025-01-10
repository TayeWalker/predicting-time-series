import os
import subprocess
import json

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

def run_multiple(docker_image):
    """Load configs and execute multiple runs."""
    with open("config.json", "r") as f:
        config = json.load(f)


    for i, run_config in enumerate(config["runs"]):
        print(f"Running configuration {i + 1}/{len(config['runs'])}: {run_config}")
        run_docker_with_params(run_config, docker_image, f"run_{i + 1}")

def run_docker_with_params(config, docker_image, run_name):
    """Run Docker container with parameters."""
    # Save temp config file for this run
    temp_config_path = os.path.join("/tmp", f"temp_config_{run_name}.json")
    with open(temp_config_path, "w") as temp_config:
        json.dump(config, temp_config)

    folder_path = multivariate_path if docker_image == docker_images["multivariate"] else univariate_path

    # Run Docker container with mounted temp config
    subprocess.run(
        [
            "docker", "run", 
            "-v", f"{temp_config_path}:/app/config.json",  # Mount temp config
            "-v", f"{folder_path}:/app",  # Mount repository
            docker_image
        ],
        check=True
    )

if __name__ == "__main__":
    build_docker("multivariate")
    run_multiple(docker_images["multivariate"])
