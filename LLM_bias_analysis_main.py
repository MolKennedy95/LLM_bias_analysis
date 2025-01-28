#from LLM_bias_analysis_functions import process_all_files_in_directory #llama_output    # process_all_files_in_directory
from LLM_preference_functions import qwen_preference_three_responses, qwen_preference, llama_preference_three_responses, llama_preference  #, qwen_preference

import pandas as pd



# process_all_files_in_directory(
#     folder_path="/mounts/data/proj/molly/LLM_bias_analysis/LLM_Topics",
#     model_name="Qwen/Qwen2.5-1.5B-Instruct",
#     hf_token="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"
# )




# llama_output(
#     folder_path="/mounts/data/proj/molly/LLM_bias_analysis/LLM_Topics",
#     model_name="meta-llama/Llama-3.2-1B-Instruct",
#     hf_token="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"
# )



# qwen_preference(
#     folder_path="/mounts/data/proj/molly/LLM_bias_analysis/LLM_Topics/QWEN_model",
#     model_name="Qwen/Qwen2.5-1.5B-Instruct",
#     hf_token="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"
# )


llama_preference(
    directory_path="/mounts/data/proj/molly/LLM_bias_analysis/LLM_Topics/LLAMA_model",
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    hf_token="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"
)




qwen_preference(
    directory_path="/mounts/data/proj/molly/LLM_bias_analysis/LLM_Topics/QWEN_model",
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    hf_token="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"
)



llama_preference_three_responses(
    directory_path="/mounts/data/proj/molly/LLM_bias_analysis/LLM_Topics/LLAMA_model",
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    hf_token="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"
)




qwen_preference_three_responses(
    directory_path="/mounts/data/proj/molly/LLM_bias_analysis/LLM_Topics/QWEN_model",
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    hf_token="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"
)




# qwen_preference_single_file(
#     file_path="/mounts/data/proj/molly/LLM_bias_analysis/LLM_Topics/QWEN_model/qwen_Social_Justice_and_Civil_Rights.xlsx",  # Replace with your test file path
#     model_name="Qwen/Qwen2.5-1.5B-Instruct",
#     hf_token="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"
# )