import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.read_csv("wandb_results_PM.csv")
df = results[results['State'] == 'finished']

# columns to keep 
views = ["m0", "m1", "m2", "m3", "m4"]
relationships = []
for view_a in views:
  for view_b in views:
    relationships.append(view_a + "_to_" + view_b)
  
coherence = ["final_scores/coherence/" + rel for rel in relationships]
col_names = ["final_scores/rec_loss"]
subset = df[
    ["model.name", "model.aggregation", "model.alpha_scalar"] + [col for col in df.columns if "final_scores" in col] 
] 
# old_name = subset['model.name'] + subset['model.aggregation'] + [str(x) for x in subset['model.alpha_scalar']]
old_name = subset['model.name'] + subset['model.aggregation']
# subset = subset[np.logical_or(
#   subset["model.alpha_scalar"] == 0.9,
#   subset["model.alpha_scalar"] == 0
#   )]
new_name = old_name.str.replace("jointavg", "AVG", regex=False)
new_name = new_name.str.replace("jointprior", "JP", regex=False)
new_name = new_name.str.replace("joint", "", regex=False)
new_name = new_name.str.replace("avg", "", regex=False)
new_name = new_name.str.replace("moe", "MoE", regex=False)
new_name = new_name.str.replace("mopoe", "MoPoE", regex=False)
new_name = new_name.str.replace("mixedprior", "MMVM", regex=False)
new_name = new_name.str.replace("unimodal", "independent", regex=False)
subset.loc[:, 'model.name'] = new_name

subset["final_scores/coherence/average"] = subset[coherence]

views = ["m0", "m1", "m2", "m3", "m4"]
relationships = []
indiv_results = []
one_to_one = []
column_names = []
for view_a in views:
  view_a_list = []
  one_to_one.append(view_a + "_to_" + view_a)
  for view_b in views:
    relationships.append(view_a + "_to_" + view_b)
    if view_a != view_b:
      view_a_list.append(view_b + "_to_" + view_a)
  indiv_results.append(view_a_list) # store view_a list
  column_names.append(str(view_a))
  indiv_results.append([view + "_cov" for view in view_a_list])
  column_names.append(str(view_a) + "_cov")
    
indiv_results.append(one_to_one)
column_names.append("one_to_one")
indiv_results.append([rel + "_cov" for rel in one_to_one])
column_names.append("one_to_one_cov")

subset[["final_scores/coherence/" + rel for rel in indiv_results[0]]].mean(axis=1)
# calculate averages
for i, cols in enumerate(indiv_results): 
  subset[column_names[i]] = subset[["final_scores/coherence/" + rel for rel in cols]].mean(axis=1)



relationships_full = ["final_scores/coherence/" + rel for rel in relationships]
cov_better = []
for rel in relationships_full:
  cov_better.append((subset[rel] < subset[rel + "_cov"]))

subset['cov_better'] = np.sum(np.array(cov_better), axis = 0)
# coherence = ["final_scores/coherence/" + rel for rel in relationships]

subset = subset.fillna(0)
df = subset[np.logical_or(
  df["model.alpha_scalar"] == 0.9,
  df["model.alpha_scalar"] == 0
  )]
df = subset[np.logical_or(
  df["model.alpha_scalar"] == 0.9,
  df["model.alpha_scalar"] == 0
  )]
metric_cols = [c for c in df.columns if "final_scores/coherence/m" in c]
metric_cols_cov = [c for c in metric_cols if "cov" in c]
metric_cols_no_cov = [c for c in metric_cols if "cov" not in c]

# 1️⃣ Average over repeats for each model
avg_df = df.groupby("model.name")[metric_cols_cov].mean()

# 2️⃣ Convert the averaged values into a square matrix
# Example: column name "final_scores/coherence/m0_to_m1" -> from=0, to=1
avg_long = avg_df.mean(axis=0).to_frame("value")  # average across models if multiple
avg_reset = avg_long.reset_index().rename(columns={"index":"metric", 0:"value"})
# avg_reset now has a 'metric' column with names like "final_scores/coherence/m0_to_m1"
avg_reset["from"], avg_reset["to"] = avg_reset["metric"].str.extract(r"m(\d+)_to_m(\d+)").astype(int).T.values

heatmap_data = avg_reset.pivot(index="from", columns="to", values="value")


data = heatmap_data.values
x_labels = heatmap_data.columns
y_labels = heatmap_data.index

# --- Plot with Matplotlib ---
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(data, cmap="viridis")

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Average Coherence", rotation=-90, va="bottom")

# Set tick labels
ax.set_xticks(np.arange(len(x_labels)))
ax.set_yticks(np.arange(len(y_labels)))
ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)
ax.set_xlabel("To model")
ax.set_ylabel("From model")
plt.title("Average Coherence (mX → mY)")
# 3️⃣ Plot the heatmap
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Annotate each cell with its value
for i in range(len(y_labels)):
    for j in range(len(x_labels)):
        text = ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", color="w")

plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Select metric columns ---
metric_cols = [c for c in df.columns if "final_scores/coherence/m" in c]
metric_cols_cov = [c for c in metric_cols if "cov" in c]
metric_cols = [c for c in metric_cols if "cov" not in c]

# --- Compute averages over repeats for each model ---
model_means = df.groupby("model.name")[metric_cols].mean()

# --- Extract all coherence pair indices (mX_to_mY) ---
def extract_matrix(avg_series):
    """Convert a row of averaged metrics into a square DataFrame."""
    tmp = avg_series.to_frame("value")
    ex = tmp.index.to_series().str.extract(r"m(\d+)_to_m(\d+)")
    tmp["from"] = ex[0].astype(int)
    tmp["to"]   = ex[1].astype(int)
    tmp = tmp.dropna(subset=["from", "to"])
    return tmp.pivot(index="from", columns="to", values="value")

# --- Build one heatmap per model ---
matrices = {name: extract_matrix(row) for name, row in model_means.iterrows()}

# --- Plot them side by side ---
n_models = len(matrices)
n_cols = min(3, n_models)  # up to 3 per row
n_rows = int(np.ceil(n_models / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = np.array(axes).reshape(-1)  # flatten in case of 1 row

for ax, (name, mat) in zip(axes, matrices.items()):
    data = mat.values
    im = ax.imshow(data, cmap="viridis")
    ax.set_title(f"{name} — Average Coherence", fontsize=12)
    ax.set_xlabel("To model")
    ax.set_ylabel("From model")
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_xticklabels(mat.columns)
    ax.set_yticklabels(mat.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)

# Add a shared colorbar
fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label="Average Coherence")

# Hide unused subplots (if any)
for ax in axes[len(matrices):]:
    ax.axis("off")

plt.tight_layout()
plt.show()


def coherence_plots(df, input):
  # Keep only the relevant columns
  cols = [
      "final_scores/coherence/m0_to_m",
      "final_scores/coherence/m1_to_m",
      "final_scores/coherence/m2_to_m",
      "final_scores/coherence/m3_to_m",
      "final_scores/coherence/m4_to_m",
  ]
  cols = [col + str(input) + "_cov" for col in cols]
  cols.append("model.name")
  df_subset = df[cols]

  # Melt into long format for grouped plotting
  df_long = df_subset.melt(id_vars="model.name", 
                          var_name="metric", 
                          value_name="score")
  df_long["metric"] = df_long["metric"].str.replace("final_scores/coherence/", "", regex=False)

  df_long["metric"] = df_long["metric"].str.replace("_to_m" + str(input), "", regex=False)
  df_long["metric"] = df_long["metric"].str.replace("m", "M", regex=False)
  # Compute mean and SEM by model + metric
  summary = (
      df_long.groupby(["model.name", "metric"])
      .agg(mean=("score", "mean"), sem=("score", "sem"))
      .reset_index()
  )

  # Pivot so we can plot grouped bars
  metrics = summary["metric"].unique()
  models = summary["model.name"].unique()
  # x = summary["model.name"].unique()
  x = np.arange(len(metrics)) 
  # bar_width = 0.25
  bar_width = 0.8 / len(models) 

  fig, ax = plt.subplots(figsize=(10,6))

  colors = plt.cm.tab10.colors[:len(models)] # type: ignore
  # Plot each metric as a group
  for i, model in enumerate(models):
      data = summary[summary["model.name"] == model]
      ax.bar(
          x + i * bar_width,
          data.set_index("metric").loc[metrics, "mean"],
          yerr=data.set_index("metric").loc[metrics, "sem"],
          color=colors[i],
          width=bar_width,
          label=model,
          capsize=4
      )

  # Formatting
  ax.set_xticks(x + bar_width * (len(models)-1)/2)
  ax.set_xticklabels(metrics)
  ax.set_ylabel("Mean Score (± SE)")
  ax.set_xlabel("Input")
  ax.set_title("Coherence Scores for M" + str(input))
  ax.legend(title="Model")

  plt.tight_layout()
  filepath = "results_figures/coherence_m" + str(input) + ".png"
  plt.savefig(filepath)
  
[coherence_plots(subset, i) for i in range(5)]


  



