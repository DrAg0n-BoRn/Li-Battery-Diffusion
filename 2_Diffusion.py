# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: li-diffusion
#     language: python
#     name: python3
# ---

# %%
from ml_tools.ML_datasetmaster import DragonDataset as ChosenDataset
from ml_tools.ML_trainer import DragonTabularDiTTrainer as ChosenTrainer
from ml_tools.ML_models_diffusion import DragonDiTGuided as ChosenModel
from ml_tools.ML_configuration import (
    FormatTabularDiffusionMetrics as ChosenMetricsConfig, 
    FinalizeTabularDiffusion as ChosenFinalizer, 
    DragonDiTParams as ChosenModelParams,    
)

from ml_tools.ML_configuration import DragonTrainingConfig
from ml_tools.ML_callbacks import DragonModelCheckpoint, DragonPatienceEarlyStopping, DragonPlateauScheduler
from ml_tools.ML_utilities import build_optimizer_params, inspect_model_architecture, DragonArtifactFinder
from ml_tools.utilities import load_dataframe_with_schema
from ml_tools.IO_tools import train_logger
from ml_tools.schema import FeatureSchema
from ml_tools.keys import TaskKeys
from torch.optim import AdamW

# Autoencoder for DIT
from ml_tools.ML_models_diffusion import DragonAutoencoder

from paths import PM
from helpers.constants import TARGET_capacity as TARGET, EMBEDDING_DIMENSION

# %%
SCHEMA_PATH = PM.schema_dir
TRAIN_DATASET_FILE = PM.processed_data_file
TRAIN_ARTIFACTS_DIR = PM.diffusion

# %% [markdown]
# ## 1. Config

# %%
train_config = DragonTrainingConfig(
    validation_size=0.1,
    test_size=0.1,
    initial_learning_rate=0.001,
    batch_size=64,
    task = TaskKeys.DIFFUSION,
    device = "cuda:0",
    finalized_filename = "guided_DIT",
    random_state=101,
    
    targets=TARGET,
    weight_decay=0.001,
    early_stop_patience=25,
    scheduler_patience=4,
    scheduler_lr_factor=0.5,
    monitor_metric="Validation Loss"
)

# %% [markdown]
# ## 2. Load Schema and Dataframe

# %%
schema = FeatureSchema.from_json(SCHEMA_PATH)

df, _ = load_dataframe_with_schema(df_path=TRAIN_DATASET_FILE, schema=schema)

# %% [markdown]
# ## 3. Make Datasets

# %%
dataset = ChosenDataset(pandas_df=df,
                        schema=schema,
                        kind=train_config.task, # type: ignore
                        feature_scaler="fit",
                        target_scaler="fit",
                        validation_size=train_config.validation_size,
                        test_size=train_config.test_size,
                        random_state=train_config.random_state,
                        )

# %% [markdown]
# ## 4. Model and Trainer

# %%
# Load autoencoder
artifact_finder = DragonArtifactFinder(directory=PM.autoencoder, load_scaler=True, load_schema=False, strict=True)

tokenizer = DragonAutoencoder.from_artifact_finder(artifact_finder)

# %%
model_params = ChosenModelParams(
    embed_dim=EMBEDDING_DIMENSION,
    seq_len=schema.number_of_features(),
    num_heads=4,
    depth=4
)

model = ChosenModel(**model_params)


# optimizer
optim_params = build_optimizer_params(model=model, weight_decay=train_config.weight_decay)
optimizer = AdamW(params=optim_params, lr=train_config.initial_learning_rate)


trainer = ChosenTrainer(model=model,
                        token_embedder=tokenizer,
                        train_dataset=dataset.train_dataset,
                        validation_dataset=dataset.validation_dataset,
                        save_dir=TRAIN_ARTIFACTS_DIR,
                        optimizer=optimizer,
                        device=train_config.device,
                        checkpoint_callback=DragonModelCheckpoint(monitor=train_config.monitor_metric),
                        early_stopping_callback=DragonPatienceEarlyStopping(patience=train_config.early_stop_patience, 
                                                                            monitor=train_config.monitor_metric),
                        lr_scheduler_callback=DragonPlateauScheduler(monitor=train_config.monitor_metric,
                                                                     patience=train_config.scheduler_patience,
                                                                     factor=train_config.scheduler_lr_factor),  
                        )

# %% [markdown]
# ## 5. Training

# %%
history = trainer.fit(epochs=500, batch_size=train_config.batch_size)

# %% [markdown]
# ## 6. Evaluation

# %%
trainer.evaluate(model_checkpoint="best",
                test_data=dataset.test_dataset,
                val_format_configuration=ChosenMetricsConfig(),
                test_format_configuration=ChosenMetricsConfig(real_color="tab:green", gen_color="violet"),
                )

# %% [markdown]
# ## 7. Save artifacts

# %%
# Dataset artifacts
dataset.save_artifacts(TRAIN_ARTIFACTS_DIR)

# Model artifacts
model.save_architecture(TRAIN_ARTIFACTS_DIR)
inspect_model_architecture(model=model, save_dir=TRAIN_ARTIFACTS_DIR)

# FeatureSchema
schema.to_json(TRAIN_ARTIFACTS_DIR)

# Train log
train_logger(train_config=train_config,
             model_parameters=model_params,
             train_history=history,
             save_directory=TRAIN_ARTIFACTS_DIR)

# %% [markdown]
# ## 8. Finalize Deep Learning

# %%
trainer.finalize_model_training(model_checkpoint='current',
                                finalize_config=ChosenFinalizer(filename=train_config.finalized_filename))
