import torch

from ml_tools.ML_models_diffusion import DragonAutoencoder, DragonDiTGuided
from ml_tools.ML_utilities import DragonArtifactFinder
from ml_tools.utilities import save_dataframe_filename
from ml_tools.path_manager import sanitize_filename
from ml_tools.math_utilities import handle_negative_values, round_float_values
from ml_tools.data_exploration import plot_value_distributions, plot_numeric_overview_boxplot

from paths import PM
from helpers.constants import TARGET_capacity as TARGET, EXPERIMENTAL_CAPACITY_RANGE

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# Hyperparameters for generation
TARGET_RANGE = range(EXPERIMENTAL_CAPACITY_RANGE[0], EXPERIMENTAL_CAPACITY_RANGE[1] + 1, 10)
GENERATION_BATCH_SIZE = 1000
GUIDANCE_SCALE = 3.0


def create_sample_batch(guided_dit: DragonDiTGuided, autoencoder: DragonAutoencoder, target_value: float, guidance_scale: float):
    
    generated_batch = guided_dit.generate_sequence(batch_size=GENERATION_BATCH_SIZE, 
                                                   target_value=target_value, 
                                                   guidance_scale=guidance_scale)
    
    # Decode generated samples
    decoded_samples = autoencoder.approximate_decode(generated_batch)
    
    # handle negative values in decoded samples (if any) by setting them to zero
    decoded_samples = handle_negative_values(df=decoded_samples)
    # round float values to N decimal places
    decoded_samples = round_float_values(df=decoded_samples, n=3)

    # batch info
    batch_info = f"{sanitize_filename(TARGET)}-{target_value}-guidance-{guidance_scale}".replace('.', '_')
    
    save_directory = PM.generation / batch_info

    # Save the decoded samples
    save_dataframe_filename(df=decoded_samples, 
                            save_dir=save_directory,
                            filename=f"generated-{GENERATION_BATCH_SIZE}",
                            verbose=1)
    
    plot_value_distributions(df=decoded_samples, save_dir=save_directory)
    
    plot_numeric_overview_boxplot(df=decoded_samples, 
                                  save_dir=save_directory, 
                                  plot_title="Generated Distribution (Scaled) - Capacity " + str(target_value) + " mAh/g", 
                                  strategy="scale", 
                                  handle_zero_variance="constant")
    
    plot_numeric_overview_boxplot(df=decoded_samples, 
                                  save_dir=save_directory, 
                                  plot_title="Generated Distribution (Log) - Capacity " + str(target_value) + " mAh/g", 
                                  strategy="log")


def main():
    # Load trained autoencoder
    autoencoder_artifacts = DragonArtifactFinder(directory=PM.autoencoder, load_scaler=True, load_schema=False, strict=True)
    autoencoder = DragonAutoencoder.from_artifact_finder(autoencoder_artifacts).to(DEVICE)
    
    # Load trained DiT
    dit_artifacts = DragonArtifactFinder(directory=PM.diffusion, load_scaler=True, load_schema=False, strict=True)
    guided_dit = DragonDiTGuided.from_artifact_finder(dit_artifacts).to(DEVICE)
    
    # Generate new samples in loops
    for TARGET_VALUE in TARGET_RANGE:
        create_sample_batch(guided_dit=guided_dit, 
                        autoencoder=autoencoder, 
                        target_value=TARGET_VALUE, 
                        guidance_scale=GUIDANCE_SCALE)


if __name__ == "__main__":
    main()
