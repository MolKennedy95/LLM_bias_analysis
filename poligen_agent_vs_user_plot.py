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
soc_User = [0.49, 0.59, 0.67, 0.49, 0.45, 0.47, 0.48] # dem
cap_User = [0.51, 0.41, 0.33, 0.51, 0.55, 0.52, 0.52] # rep

# Data for Agent mode
soc_agent = [0.45, 0.51, 0.49, 0.48, 0.49, 0.48, 0.49]
cap_agent = [0.55, 0.49, 0.51, 0.5 , 0.51, 0.52, 0.51]

# Positions for each group (one group per model)
x = np.arange(len(model_names))  # [0, 1, 2, 3]
bar_width = 0.3

fig, ax = plt.subplots(figsize=(10, 6))

# --- Plot User bars (stacked) ---
# Democratic portion for User
bars_User_dem = ax.bar(
    x - bar_width / 2, soc_User, bar_width, color="blue", alpha=1.0, label="User (Soc)"
)

# Republican portion stacked on top (User)
bars_User_rep = ax.bar(
    x - bar_width / 2,
    cap_User,
    bar_width,
    bottom=soc_User,
    color="red",
    alpha=1.0,
    label="User (Cap)",
)

# --- Plot Agent bars (stacked) ---
# Democratic portion for Agent
bars_agent_dem = ax.bar(
    x + bar_width / 2,
    soc_agent,
    bar_width,
    color="blue",
    alpha=0.5,
    label="Agent (Soc)",
)

# Republican portion stacked on top (Agent)
bars_agent_rep = ax.bar(
    x + bar_width / 2,
    cap_agent,
    bar_width,
    bottom=soc_agent,
    color="red",
    alpha=0.5,
    label="Agent (Cap)",
)

# Set x-axis tick positions and labels
ax.set_xticks(x)
ax.set_xticklabels(model_names)

# Labels and title
ax.set_ylabel("Proportion")
ax.set_title("Socialist vs Capitalist Proportions by Model (User vs Agent)")

# Only show each label once in the legend
# (Matplotlib would otherwise generate duplicates since we used the same colors multiple times.)
handles, labels = ax.get_legend_handles_labels()
unique = list(dict(zip(labels, handles)).items())  # preserves order, removes duplicates
ax.legend([handle for _, handle in unique], [label for label, _ in unique])

plt.tight_layout()
plt.show()
