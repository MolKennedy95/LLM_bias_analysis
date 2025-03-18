import matplotlib.pyplot as plt
import numpy as np

# Model names (4 models total)
model_names = [
    "Phi",
    "GLM",
    "Qwen",
    "Mistral",
    "Command R+",
    "Llama",
    "R1",
]

# Data for User mode
dem_User = [0.76, 0.74, 0.59, 0.59, 0.52, 0.54, 0.54] # dem
rep_User = [0.24, 0.26, 0.41, 0.41, 0.48, 0.46, 0.46] # rep

# Data for Agent mode
dem_agent = [0.91, 0.81, 0.8, 0.73, 0.51, 0.52, 0.49]
rep_agent = [0.1, 0.19, 0.2, 0.27 , 0.5, 0.48, 0.51]

# Positions for each group (one group per model)
x = np.arange(len(model_names))  # [0, 1, 2, 3]
bar_width = 0.3

fig, ax = plt.subplots(figsize=(10, 6))

# --- Plot User bars (stacked) ---
# Democratic portion for User
bars_User_dem = ax.bar(
    x - bar_width / 2, dem_User, bar_width, color="blue", alpha=1.0, label="User Preference (Dem)"
)

# Republican portion stacked on top (User)
bars_User_rep = ax.bar(
    x - bar_width / 2,
    rep_User,
    bar_width,
    bottom=dem_User,
    color="red",
    alpha=1.0,
    label="User Preference (Rep)",
)

# --- Plot Agent bars (stacked) ---
# Democratic portion for Agent
bars_agent_dem = ax.bar(
    x + bar_width / 2,
    dem_agent,
    bar_width,
    color="blue",
    alpha=0.5,
    label="User Least Biased (Dem)",
)

# Republican portion stacked on top (Agent)
bars_agent_rep = ax.bar(
    x + bar_width / 2,
    rep_agent,
    bar_width,
    bottom=dem_agent,
    color="red",
    alpha=0.5,
    label="User Least Biased (Rep)",
)

# Set x-axis tick positions and labels
ax.set_xticks(x)
ax.set_xticklabels(model_names)

# Labels and title
ax.set_ylabel("Proportion")
ax.set_title("Democratic vs Republican Proportions by Model (User Preference vs User Least Biased)")

# Only show each label once in the legend
# (Matplotlib would otherwise generate duplicates since we used the same colors multiple times.)
handles, labels = ax.get_legend_handles_labels()
unique = list(dict(zip(labels, handles)).items())  # preserves order, removes duplicates
ax.legend([handle for _, handle in unique], [label for label, _ in unique])

plt.tight_layout()

plt.savefig("poligen_user_preference_vs_user_least_biased.png", dpi=300, bbox_inches="tight")  # Save as PNG with high resolution

plt.show()
