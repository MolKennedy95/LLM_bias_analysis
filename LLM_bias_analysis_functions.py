from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import pandas as pd
import chardet



# ### TEST FUNCTION:

# def qwen_output_single_file(file_path, model_name, hf_token):
#     # Load model and tokenizer
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype="auto",
#         device_map="cuda:5",
#         cache_dir="/mounts/data/proj/molly/cache",
#         token=hf_token,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

#     # Read the Excel file
#     try:
#         df = pd.read_excel(file_path, engine='openpyxl')
#     except Exception as e:
#         print(f"Error reading {file_path}: {e}")
#         return

#     # Ensure the 'Topic' column exists
#     if 'Topic' not in df.columns:
#         print(f"Skipping {file_path}: 'Topic' column not found.")
#         return

#     # Initialize new columns for articles
#     df['Democratic_article'] = ''
#     df['Republican_article'] = ''
#     df['Neutral_article'] = ''

#     for index, row in df.iterrows():
#         topic = row['Topic']
#         for wing in ["democratic", "republican", "neutral"]:
#             prompt = f" write a {wing} article on {topic}. Begin with: This article is about {topic}, followed by the article title and the article."

#             # Qwen template
#             messages = [
#                 {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ]

#             text = tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )

#             model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#             generated_ids = model.generate(
#                 **model_inputs,
#                 max_new_tokens=512
#             )

#             generated_ids = [
#                 output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#             ]

#             response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

#             # Append response to the appropriate column
#             if wing == "democratic":
#                 df.at[index, 'Democratic_article'] = response
#             elif wing == "republican":
#                 df.at[index, 'Republican_article'] = response
#             elif wing == "neutral":
#                 df.at[index, 'Neutral_article'] = response

#     # Save updated DataFrame back to Excel
#     try:
#         df.to_excel(file_path, index=False, engine='openpyxl')
#         print(f"Updated file saved: {file_path}")
#     except Exception as e:
#         print(f"Error saving {file_path}: {e}")

# def test_single_file():
#     file_path = "/mounts/data/proj/molly/LLM_bias_analysis/LLM_Topics/Social_Justice_and_Civil_Rights.xlsx"  # Replace with your test file
#     model_name = "Qwen/Qwen2.5-1.5B-Instruct"
#     hf_token = "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"
#     qwen_output_single_file(file_path, model_name, hf_token)

# # Test the function on a single file
# test_single_file()



import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

def qwen_output_single_file(file_path, model_name, hf_token):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda:5",
        cache_dir="/mounts/data/proj/molly/cache",
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # Read the Excel file
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Ensure the 'Topic' column exists
    if 'Topic' not in df.columns:
        print(f"Skipping {file_path}: 'Topic' column not found.")
        return

    # Initialize new columns for articles
    df['Democratic_article'] = ''
    df['Republican_article'] = ''
    df['Neutral_article'] = ''

    for index, row in df.iterrows():
        topic = row['Topic']
        for wing in ["democratic", "republican", "neutral"]:
            prompt = f" write a {wing} article on {topic}. Begin with: This article is about {topic}, followed by the article title and the article."

            # Qwen template
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": prompt}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Append response to the appropriate column
            if wing == "democratic":
                df.at[index, 'Democratic_article'] = response
            elif wing == "republican":
                df.at[index, 'Republican_article'] = response
            elif wing == "neutral":
                df.at[index, 'Neutral_article'] = response

    # Save updated DataFrame back to Excel
    try:
        df.to_excel(file_path, index=False, engine='openpyxl')
        print(f"Updated file saved: {file_path}")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")

def process_all_files_in_directory(folder_path, model_name, hf_token):
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx"):  # Process only .xlsx files
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            qwen_output_single_file(file_path, model_name, hf_token)

# # Example usage
# process_all_files_in_directory(
#     folder_path="/mounts/data/proj/molly/LLM_bias_analysis/LLM_Topics",
#     model_name="Qwen/Qwen2.5-1.5B-Instruct",
#     hf_token="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"
# )






### Llama model


import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

def llama_output(folder_path, model_name, hf_token):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda:5",
        cache_dir="/mounts/data/proj/molly/cache",
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # Iterate over xlsx files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(folder_path, filename)
            
            # Read the Excel file
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            # Ensure the 'Topic' column exists
            if 'Topic' not in df.columns:
                print(f"Skipping {file_path}: 'Topic' column not found.")
                continue

            # Initialize new columns for articles
            df['Democratic_article'] = ''
            df['Republican_article'] = ''
            df['Neutral_article'] = ''

            for index, row in df.iterrows():
                topic = row['Topic']

                for wing in ["democratic", "republican", "neutral"]:
                    # Prepare the prompt
                    prompt = f" write a {wing} article on {topic}. Begin with: This article is about {topic}, followed by the article title and the article."

                    llama_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

                    text = llama_template.format(prompt)

                    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=512
                    )

                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    # Append response to the appropriate column
                    if wing == "democratic":
                        df.at[index, 'Democratic_article'] = response
                    elif wing == "republican":
                        df.at[index, 'Republican_article'] = response
                    elif wing == "neutral":
                        df.at[index, 'Neutral_article'] = response

            # Save updated DataFrame back to Excel
            try:
                df.to_excel(file_path, index=False, engine='openpyxl')
                print(f"Updated file saved: {file_path}")
            except Exception as e:
                print(f"Error saving {file_path}: {e}")

# # Example usage
# llama_output(
#     folder_path="/mounts/data/proj/molly/LLM_bias_analysis/LLM_Topics",
#     model_name="meta-llama/Llama-3.2-1B-Instruct",
#     hf_token="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"
# )
