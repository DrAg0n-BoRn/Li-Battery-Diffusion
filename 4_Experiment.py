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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from ml_tools.data_exploration import (plot_numeric_overview_boxplot_macro, 
                                       plot_value_distributions_multi, 
                                       filter_subset_continuous, 
                                       reconstruct_from_schema,
                                       summarize_dataframe)
from ml_tools.utilities import load_dataframe
from ml_tools.schema import FeatureSchema
from ml_tools.plot_fonts import  configure_cjk_fonts

from helpers.constants import TARGET_capacity as TARGET, EXPERIMENTAL_CAPACITY_RANGE, CONTINUOUS_RANGE
from paths import PM

configure_cjk_fonts()

# %% [markdown]
# ## Training data overview

# %%
df_train, _ = load_dataframe(PM.processed_data_file)

# %%
# Select subset where Target is between the experimental range, and drop the target column
df_train_filter = filter_subset_continuous(df=df_train, 
                                           range_filters={TARGET: EXPERIMENTAL_CAPACITY_RANGE},
                                           drop_filter_cols=True)

# %%
#reconstruct categorical columns from the schema
schema = FeatureSchema.from_json(PM.schema_dir)

df_train_reconstructed = reconstruct_from_schema(df=df_train_filter, schema=schema)

# %%
summarize_dataframe(df_train_reconstructed)

# %%
# Plots
train_save_dir = PM.experiment / "Train Data Plots"

plot_numeric_overview_boxplot_macro(df=df_train_reconstructed,
                                    save_dir=train_save_dir,
                                    plot_title=f"Training Data Distribution - Capacity {EXPERIMENTAL_CAPACITY_RANGE[0]}-{EXPERIMENTAL_CAPACITY_RANGE[1]} mAh/g",
                                    handle_zero_variance="constant",
                                    font_scaling=1.5)

# %% [markdown]
# ## Experimental data overview

# %%
df_experiment_raw, _ = load_dataframe(PM.experiment_data_file)

# %%
# use columns present in the training data for the experimental data overview, drop constant categorical columns
numerical_columns = [col for col in df_experiment_raw.columns if col in CONTINUOUS_RANGE.keys()]
df_experiment = df_experiment_raw[numerical_columns].astype(float)

# %%
summarize_dataframe(df_experiment)

# %%
# Plots
experiment_save_dir = PM.experiment / "Experiment Data Plots"

plot_numeric_overview_boxplot_macro(df=df_experiment, 
                                    save_dir=experiment_save_dir,
                                    plot_title=f"Experimental Data Distribution - Capacity {EXPERIMENTAL_CAPACITY_RANGE[0]}-{EXPERIMENTAL_CAPACITY_RANGE[1]} mAh/g",
                                    handle_zero_variance="constant",
                                    font_scaling=1.5)

# %% [markdown]
# ## Comparison plot

# %%
df_generated, _ = load_dataframe(PM.experiment / "generated-250.csv") # Must be placed manually in the directory after generation, as it is an output file.

# %%
summarize_dataframe(df_generated)

# %%
named_dataframes_full = {f"Train Data ({EXPERIMENTAL_CAPACITY_RANGE[0]}-{EXPERIMENTAL_CAPACITY_RANGE[1]} mAh/g)": df_train_reconstructed, 
                        "Generated Data (250 mAh/g)": df_generated}

named_dataframes_continuous = {f"Train Data ({EXPERIMENTAL_CAPACITY_RANGE[0]}-{EXPERIMENTAL_CAPACITY_RANGE[1]} mAh/g)": df_train_reconstructed[numerical_columns], 
                                "Generated Data (250 mAh/g)": df_generated[numerical_columns],
                                f"Experimental Data ({EXPERIMENTAL_CAPACITY_RANGE[0]}-{EXPERIMENTAL_CAPACITY_RANGE[1]} mAh/g)": df_experiment}

# %%
# generate comparison plots between training and generated data for all features, including categorical columns
plot_value_distributions_multi(named_dataframes=named_dataframes_full,
                               save_dir=PM.experiment,
                               font_scaling=1.2,
                               mode="percentage")

# %%
# overwrite continuous plots using the 3 datasets
plot_value_distributions_multi(named_dataframes=named_dataframes_continuous,
                               save_dir=PM.experiment,
                               font_scaling=1.2,
                               mode="percentage")
