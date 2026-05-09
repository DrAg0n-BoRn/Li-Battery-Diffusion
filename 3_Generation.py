import torch

from ml_tools.ML_models_diffusion import DragonAutoencoder, DragonDiTGuided
from ml_tools.ML_utilities import DragonArtifactFinder
from ml_tools.ML_inference_diffusion import DragonDiTGuidedGenerator

from paths import PM
from helpers.constants import EXPERIMENTAL_CAPACITY_RANGE

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# Hyperparameters for generation
TARGET_RANGE = list(range(EXPERIMENTAL_CAPACITY_RANGE[0], EXPERIMENTAL_CAPACITY_RANGE[1] + 1, 10))
GENERATION_BATCH_SIZE = 500
GUIDANCE_SCALE = 3.0


def main():
    # Load trained autoencoder
    autoencoder_artifacts = DragonArtifactFinder(directory=PM.autoencoder, load_scaler=True, load_schema=False, strict=True)
    autoencoder = DragonAutoencoder.from_artifact_finder(autoencoder_artifacts).to(DEVICE)
    
    # Load trained DiT
    dit_artifacts = DragonArtifactFinder(directory=PM.diffusion, load_scaler=True, load_schema=False, strict=True)
    guided_dit = DragonDiTGuided.from_artifact_finder(dit_artifacts).to(DEVICE)
    
    generator = DragonDiTGuidedGenerator(save_dir=PM.generation,
                                         diffusion_model=guided_dit,
                                         encoder=autoencoder,
                                         device=DEVICE)
    
    generator.generate_plot_multi(targets=TARGET_RANGE, # type: ignore
                                  batch_size=GENERATION_BATCH_SIZE,
                                  guidance_scale=GUIDANCE_SCALE,
                                  ode_steps=30,
                                  positive_columns="all",
                                  round_float_columns="all",
                                  float_rounding_precision=3,
                                  handle_zero_variance="constant",
                                  font_scaling=1.5)


if __name__ == "__main__":
    main()
