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
from ml_tools.data_exploration import (summarize_dataframe,
                                       show_null_columns,
                                       plot_value_distributions,
                                       plot_numeric_overview_boxplot)
from ml_tools.outlier_detection import clip_outliers_multi
from ml_tools.utilities import save_dataframe_with_schema, load_dataframe
from ml_tools.schema import FeatureSchema

from paths import PM
from helpers.constants import TARGET_capacity_retention, TARGET_first_coulombic_eff, CONTINUOUS_RANGE

# %% [markdown]
# ## Load dataframe

# %%
df_original, _ = load_dataframe(df_path=PM.original_data_file)
# drop unused target columns
df_original = df_original.drop(columns=[TARGET_capacity_retention, TARGET_first_coulombic_eff])

# %%
summarize_dataframe(df_original)

# %% [markdown]
# ## Clip outliers

# %%
df_clip = clip_outliers_multi(df=df_original, clip_dict=CONTINUOUS_RANGE)

# %%
summarize_dataframe(df_clip)

# %%
show_null_columns(df_clip)

# %% [markdown]
# ## Plot distributions

# %%
# select numerical columns for distribution plots
df_numerical = df_clip[CONTINUOUS_RANGE.keys()]

# %%
plot_value_distributions(df=df_numerical, save_dir=PM.engineering)

# %%
for _strategy in ["scale", "log"]:
    plot_numeric_overview_boxplot(df=df_numerical, 
                                  save_dir=PM.engineering, 
                                  plot_title=f"Training Data Overview ({_strategy.capitalize()})", 
                                  strategy=_strategy,  # type: ignore
                                  handle_zero_variance="constant")

# %% [markdown]
# ## Save processed dataframe

# %%
schema = FeatureSchema.from_json(PM.schema_dir)

# %%
save_dataframe_with_schema(df=df_clip, full_path=PM.processed_data_file, schema=schema)
