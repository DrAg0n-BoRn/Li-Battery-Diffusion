from ml_tools.path_manager import DragonPathManager


# 1. Initialize the PathManager using this file as the anchor, adding base directories.
PM = DragonPathManager(
    anchor_file=__file__,
    base_directories=["helpers", "start_data", "results", "backups"]
)

# 2. Define directories and files.
### Start files - From Lithium Battery optimization project
PM.original_data_file = PM.start_data / "original_data.csv"
PM.schema_dir = PM.start_data / "Schema"

### Feature Engineering
PM.engineering = PM.results / "Feature Engineering"
PM.processed_data_file = PM.engineering / "processed_data.csv"

### Autoencoder
PM.autoencoder = PM.results / "Autoencoder"

### Diffusion
PM.diffusion = PM.results / "Diffusion"
PM.generation = PM.results / "Diffusion Generation"

### Experimental data
PM.experiment = PM.results / "Experiment"
PM.experiment_data_file = PM.start_data / "experiment_data.csv"


# 3. Make directories and check status
PM.make_dirs()

if __name__ == "__main__":
    PM.status()
