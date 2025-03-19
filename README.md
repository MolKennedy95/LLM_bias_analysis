# Through the LLM Looking Glass: A Socratic Self-Assessment of Donkeys, Elephants, and Markets

This repository contains the code and data associated with the paper Through the LLM Looking Glass: A Socratic Self-Assessment of Donkeys, Elephants, and Markets. The paper can be accessed at [LINK_HERE].

## Abstract

While detecting and avoiding bias in LLM-generated text is becoming increasingly important, media bias often remains subtle and subjective, making it particularly difficult to identify and mitigate. In this study, we assess media bias in LLM-generated content and LLMs’ ability to detect subtle ideological bias. We conduct this evaluation using two datasets, PoliGen and EconoLex, covering political and economic discourse, respectively. We evaluate eight widely used LLMs by prompting them to generate articles and analyze their ideological preferences via self-assessment. By using self-assessment, the study aims to directly measure the models’ biases rather than relying on external interpretations, thereby minimizing subjective judgments about media bias. Our results reveal a consistent preference of Democratic over Republican positions across all models. Conversely, in economic topics, biases vary among Western LLMs, while those developed in China lean more strongly toward socialism.

## Introduction

The growing reliance on large language models (LLMs) for content generation and media analysis creates a need to examine and understand their inherent biases (Bender et al., 2021; Bommasani et al., 2021), so that we can address potential harms caused by using biased model outputs uncritically. Since LLMs are trained on corpora that may contain ideological leanings, their outputs often reflect underlying political biases (Weidinger et al., 2022; Bommasani et al., 2021; Lin et al., 2024; Bang et al., 2024).

Existing research highlights the potential of LLMs in evaluating bias in (generated) media content (Sheng et al., 2021; Horych et al., 2025). Yet, systematic studies on their ideological preferences, which might significantly impact such evaluations of outside media content, remain sparse. Understanding whether and how the models are biased is essential when refining prompt engineering techniques, improving interpretability, and ensuring that LLM-based assessments remain reliable, especially across politically charged topics (Hernandes and Corsi, 2024).

Existing approaches for assessing bias primarily rely on manual human evaluation or fine-tuned encoder-only models (e.g., reward models). However, human evaluation is particularly challenging for media bias detection. Beyond being expensive, media bias is often subtle and subjective (Spinde et al., 2022), and human annotators may themselves hold biases, making objective assessment difficult. Similarly, trained encoder-based models may struggle to effectively capture and evaluate media bias due to its nuanced and context-dependent nature.

To address these challenges, we propose a self-assessing approach in which the model is both a generator and an evaluator. Using a Socratic method, the model generates biased content and selects its preferred response. Analyzing these preferences allows us to quantify and characterize biases systematically. Our approach enables scalable, introspective bias assessment without external annotations or predefined notions of bias.

We present a systematic study of the degree of bias in eight widely used LLMs across various political and economic topics, followed by further analysis of LLMs’ integrity and agentic behavior. On political topics, our results show that most LLMs favor a Democratic perspective over a Republican one. In economic topics, Western-developed models remain relatively neutral, whereas models developed in China lean more strongly toward a socialist perspective, complementing the findings of (Buyl et al., 2025). Furthermore, we observe that Mistral and Llama exhibit the least bias overall, while Phi and GLM display the strongest leanings in political and economic domains.

We publicly share all code and data in this repository.

For more details, please refer to the full paper at [LINK_HERE].


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
