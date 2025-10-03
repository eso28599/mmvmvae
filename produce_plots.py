import wandb
import pandas as pd

# --- Connect to W&B project ---
api = wandb.Api()
runs = api.runs("eso18-imperial-college-london/mvvae_polymnist")  # <-- change this

records = []
for run in runs:
    if "model.aggregation" in run.config:
        records.append({
            "model.aggregation": run.config["model.aggregation"],
            "m0_to_m0": run.summary.get("final_scores/coherence/m0_to_m0"),
            "m1_to_m0": run.summary.get("final_scores/coherence/m1_to_m0"),
            "m2_to_m0": run.summary.get("final_scores/coherence/m2_to_m0"),
        })

df = pd.DataFrame(records)

# --- Compute mean + SEM ---
agg_df = (
    df.groupby("model.aggregation")
      .agg(["mean", "sem"])
      .reset_index()
)

# Flatten multi-level columns
agg_df.columns = ["model.aggregation",
                  "m0_to_m0_mean", "m0_to_m0_sem",
                  "m1_to_m0_mean", "m1_to_m0_sem",
                  "m2_to_m0_mean", "m2_to_m0_sem"]

# --- Melt into long format ---
long_df = pd.DataFrame()
for metric in ["m0_to_m0", "m1_to_m0", "m2_to_m0"]:
    tmp = agg_df[["model.aggregation", f"{metric}_mean", f"{metric}_sem"]]
    tmp = tmp.rename(columns={f"{metric}_mean": "mean", f"{metric}_sem": "sem"})
    tmp["metric"] = metric
    long_df = pd.concat([long_df, tmp], ignore_index=True)

# Build wandb Table
table = wandb.Table(dataframe=long_df)

# --- Custom Vega spec for grouped bars + error bars ---
vega_spec = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "data": {"name": "table"},
    "encoding": {
        "x": {"field": "model.aggregation", "type": "nominal", "axis": {"title": "Aggregation"}},
        "y": {"field": "mean", "type": "quantitative", "axis": {"title": "Average Score"}},
        "color": {"field": "metric", "type": "nominal", "legend": {"title": "Metric"}},
    },
    "layer": [
        # Bars grouped by aggregation + metric
        {
            "mark": {"type": "bar", "tooltip": True},
            "encoding": {
                "x": {"field": "model.aggregation", "type": "nominal"},
                "y": {"field": "mean", "type": "quantitative"},
                "color": {"field": "metric", "type": "nominal"}
            }
        },
        # Error bars
        {
            "mark": {"type": "errorbar"},
            "encoding": {
                "x": {"field": "model.aggregation", "type": "nominal"},
                "y": {"field": "mean", "type": "quantitative"},
                "yError": {"field": "sem", "type": "quantitative"},
                "color": {"field": "metric", "type": "nominal"}
            }
        }
    ]
}

# --- Log the plot ---
wandb.init(project="my-project")  # start run
wandb.log({
    "Average Scores (Grouped by Aggregation)": wandb.plot_table(
        vega_spec,
        table,
        fields={"x": "model.aggregation", "y": "mean", "group": "metric"}
    )
})
