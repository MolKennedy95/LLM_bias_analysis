# Through the LLM Looking Glass: A Socratic Probing of Donkeys, Elephants, and Markets

This repository contains the code and data associated with the paper Through the LLM Looking Glass: A Socratic Probing of Donkeys, Elephants, and Markets. The paper can be accessed at https://arxiv.org/abs/2503.16674

## Abstract

While detecting and avoiding bias in LLM-generated text is becoming increasingly important, media bias often remains subtle and subjective, making it particularly difficult to identify
and mitigate. In this study, we assess media bias in LLM-generated content and LLMs' ability to detect subtle ideological bias. We conduct this evaluation using two datasets, PoliGen and
EconoLex, covering political and economic discourse, respectively. We evaluate seven widely used LLMs by prompting them to generate articles and analyze their ideological preferences via Socratic probing. 
By using our self-contained Socratic approach, the study aims to directly measure the models' biases rather than relying on external interpretations, thereby minimizing subjective judgments about media bias. 
Our results reveal a consistent preference of Democratic over Republican positions across all models. Conversely, in economic topics, biases vary among Western LLMs, while those developed in 
China lean more strongly toward socialism.

We publicly share all code and data in this repository.

For more details, please refer to the full paper at https://arxiv.org/abs/2503.16674


## Repository Structure: LLM_bias_analysis
This repository contains code, data, and scripts related to your study on bias in LLMs.

### Data Folders
- EconoLex_data/ – Contains dataset of economic related headlines from news outlets used in the study.
- PoliGen_data/ – Contains datasets of political related topics used in the study.
### Code and Functionality
- LLM_article_generation_functions/ – Contains scripts responsible for generating articles from LLMs for EconoLex and PoliGen datasets.
- LLM_article_preference_generation/ – Contains scripts for generating LLM-based preference indications for the generated articles.
- Plot_results_scripts/ – Scripts for visualizing results and generating plots.
### Results and Ratios
- micro_averaged_ratios/ – Ratios of results for PoliGen dataset.
- ratios/ – Ratios of results for EconoLex dataset.
### Evaluation Scripts
- econolex_user_preference_vs_user_least_biased.py – A script analyzing user preferences and least biased perspectives in the EconoLex dataset.
- poligen_agent_vs_user_plot.py – Generates visualizations comparing user and agent biases for PoliGen.
- poligen_user_preference_vs_user_least_biased.py – Similar to the EconoLex script but applied to PoliGen data.
