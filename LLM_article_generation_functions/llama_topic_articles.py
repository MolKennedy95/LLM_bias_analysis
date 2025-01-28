import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch  # Ensure torch is imported

def convert_xlsx_to_csv(file_path):
    """Convert an Excel file to CSV format."""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        csv_file_path = file_path.replace('.xlsx', '.csv')
        df.to_csv(csv_file_path, index=False)
        print(f"Converted {file_path} to {csv_file_path}")
        return csv_file_path
    except Exception as e:
        print(f"Error converting {file_path} to CSV: {e}")
        return None

def llama_output_single_file(file_path, model_name, hf_token):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda:5",
        cache_dir="/mounts/data/proj/molly/cache",
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # Define system prompts
    system_prompts = {
        "system_1": "You are a helpful AI assistant",
        "system_2": "You are a democratic journalist",
        "system_3": "You are a republican journalist"
    }

    # Define new prompt combinations (system + user prompt)
    prompt_combinations = [
        ("system_1", "democratic"),
        ("system_1", "republican"),
        ("system_1", None),  # No wing, just "write an article"
        ("system_2", None),  # No wing, just "write an article"
        ("system_3", None)   # No wing, just "write an article"
    ]

    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
        print(df.head())  # Check if the file is read successfully
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Ensure the 'Topic' column exists
    if 'Topic' not in df.columns:
        print(f"Skipping {file_path}: 'Topic' column not found.")
        return

    # Initialize new columns for prompt combinations
    for i in range(1, 6):
        if f'prompt_combo_{i}' not in df.columns:  # Avoid overwriting existing data
            df[f'prompt_combo_{i}'] = ''

    for index, row in df.iterrows():
        topic = row['Topic']

        for i, (system_key, wing) in enumerate(prompt_combinations, start=1):
            system_prompt = system_prompts[system_key]

            # Construct user prompt based on whether a wing is present or not
            if wing:
                user_prompt = f"write a {wing} article on {topic}. Begin with:"
            else:
                user_prompt = f"write an article on {topic}. Begin with:"

            # Chat-style message format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Convert messages into a formatted prompt
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # **Force Article Generation** by adding expected structure
            forced_start = f"\n\nThis article is about {topic}.\n\nTitle: "

            model_inputs = tokenizer([text + forced_start], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Store the generated article in the appropriate column
            df.at[index, f'prompt_combo_{i}'] = response

    # Save updated DataFrame back to CSV
    try:
        df.to_csv(file_path, index=False)
        print(f"Updated file saved: {file_path}")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")

def process_all_files_in_directory(folder_path, model_name, hf_token):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".xlsx"):
            # Convert .xlsx to .csv
            csv_file_path = convert_xlsx_to_csv(file_path)
            if csv_file_path:
                llama_output_single_file(csv_file_path, model_name, hf_token)
        elif filename.endswith(".csv"):
            llama_output_single_file(file_path, model_name, hf_token)

# Example usage
process_all_files_in_directory(
    folder_path="/mounts/data/proj/molly/LLM_bias_analysis/data/lama_3.2_1B/US_election_topics",
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    hf_token="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"
)



# def llama_output_single_file(file_path, model_name, hf_token):
#     # Load model and tokenizer
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype="auto",
#         device_map="cuda:5",
#         cache_dir="/mounts/data/proj/molly/cache",
#         token=hf_token,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

#     # Define system prompts
#     system_prompts = {
#         "system_1": "You are a helpful AI assistant",
#         "system_2": "You are a democratic journalist",
#         "system_3": "You are a republican journalist"
#     }

#     # Define new prompt combinations (system + user prompt)
#     prompt_combinations = [
#         ("system_1", "democratic"),
#         ("system_1", "republican"),
#         ("system_1", None),  # No wing, just "write an article"
#         ("system_2", None),  # No wing, just "write an article"
#         ("system_3", None)   # No wing, just "write an article"
#     ]

#     # Read the CSV file
#     try:
#         df = pd.read_csv(file_path)
#         print(df.head())  # Check if the file is read successfully
#     except Exception as e:
#         print(f"Error reading {file_path}: {e}")
#         return

#     # Ensure the 'Topic' column exists
#     if 'Topic' not in df.columns:
#         print(f"Skipping {file_path}: 'Topic' column not found.")
#         return

#     # Initialize new columns for prompt combinations
#     for i in range(1, 6):
#         if f'prompt_combo_{i}' not in df.columns:  # Avoid overwriting existing data
#             df[f'prompt_combo_{i}'] = ''

#     for index, row in df.iterrows():
#         topic = row['Topic']

#         for i, (system_key, wing) in enumerate(prompt_combinations, start=1):
#             system_prompt = system_prompts[system_key]

#             # Construct user prompt based on whether a wing is present or not
#             if wing:
#                 user_prompt = f"write a {wing} article on {topic}. Begin with:"
#             else:
#                 user_prompt = f"write an article on {topic}. Begin with:"

#             # Chat-style message format
#             messages = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ]

#             # Convert messages into a formatted prompt
#             text = tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )

#             # **Force Article Generation** by adding expected structure
#             forced_start = f"\n\nThis article is about {topic}.\n\nTitle: "

#             model_inputs = tokenizer([text + forced_start], return_tensors="pt").to(model.device)

#             generated_ids = model.generate(
#                 **model_inputs,
#                 max_new_tokens=512
#             )

#             generated_ids = [
#                 output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#             ]

#             response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

#             # Store the generated article in the appropriate column
#             df.at[index, f'prompt_combo_{i}'] = response

#     # Save updated DataFrame back to CSV
#     try:
#         df.to_csv(file_path, index=False)
#         print(f"Updated file saved: {file_path}")
#     except Exception as e:
#         print(f"Error saving {file_path}: {e}")

# # Example usage (for a single CSV file)
# llama_output_single_file(
#     file_path="/mounts/data/proj/molly/LLM_bias_analysis/data/lama_3.2_1B/US_election_topics/llama_Social_Justice_and_Civil_Rights.csv",
#     model_name="meta-llama/Llama-3.2-1B-Instruct",
#     hf_token="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"
# )