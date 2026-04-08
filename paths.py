from ml_tools.path_manager import DragonPathManager


# 1. Initialize the PathManager using this file as the anchor, adding base directories.
PM = DragonPathManager(
    anchor_file=__file__,
    base_directories=["helpers", "start_data", "results", "backups"]
)

# 2. Define directories and files.
### Base files
PM.processed_data_file = PM.start_data / "processed_data.csv" # From Lithium Battery optimization project

### Feature Engineering
PM.engineering = PM.start_data / "Feature Engineering" # From Lithium Battery optimization project

### Autoencoder
PM.autoencoder = PM.results / "Autoencoder"

### Diffusion
PM.diffusion = PM.results / "Diffusion"
PM.diffusion_generation = PM.diffusion / "Generation"


# 3. Make directories and check status
PM.make_dirs()

if __name__ == "__main__":
    PM.status()
