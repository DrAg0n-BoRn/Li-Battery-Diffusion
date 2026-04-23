---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: li-diffusion
    language: python
    name: python3
---

```python
from ml_tools.data_exploration import (summarize_dataframe,
                                       show_null_columns,
                                       plot_value_distributions,
                                       plot_numeric_overview_boxplot)
from ml_tools.outlier_detection import clip_outliers_multi
from ml_tools.utilities import save_dataframe_with_schema, load_dataframe
from ml_tools.schema import FeatureSchema

from paths import PM
from helpers.constants import TARGET_capacity_retention, TARGET_first_coulombic_eff, CONTINUOUS_RANGE
```

## Load dataframe

```python
df_original, _ = load_dataframe(df_path=PM.original_data_file)
# drop unused target columns
df_original = df_original.drop(columns=[TARGET_capacity_retention, TARGET_first_coulombic_eff])
```

```python
summarize_dataframe(df_original)
```

## Clip outliers

```python
df_clip = clip_outliers_multi(df=df_original, clip_dict=CONTINUOUS_RANGE)
```

```python
summarize_dataframe(df_clip)
```

```python
show_null_columns(df_clip)
```

## Plot distributions

```python
# select numerical columns for distribution plots
df_numerical = df_clip[CONTINUOUS_RANGE.keys()]
```

```python
plot_value_distributions(df=df_numerical, save_dir=PM.engineering)
```

```python
for _strategy in ["scale", "log"]:
    plot_numeric_overview_boxplot(df=df_numerical, 
                                  save_dir=PM.engineering, 
                                  plot_title=f"Training Data Overview ({_strategy.capitalize()})", 
                                  strategy=_strategy,  # type: ignore
                                  handle_zero_variance="constant")
```

## Save processed dataframe

```python
schema = FeatureSchema.from_json(PM.schema_dir)
```

```python
save_dataframe_with_schema(df=df_clip, full_path=PM.processed_data_file, schema=schema)
```
