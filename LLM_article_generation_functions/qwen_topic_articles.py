import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def qwen_output_single_file(file_path, model_name, hf_token):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:7",
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

    # Define prompt combinations (system + user prompt)
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
        print(f"Processing file: {file_path}")
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

            # Qwen chat-style message format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Convert messages into a formatted chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # **Force Article Generation** by adding expected structure
            forced_start = f"\n\nThis article is about {topic}.\n\nTitle: "

            model_inputs = tokenizer([text + forced_start], return_tensors="pt").to(model.device)

            try:
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            except Exception as e:
                print(f"Error during generation for topic '{topic}': {e}")
                response = ""

            # Store the generated article in the appropriate column
            df.at[index, f'prompt_combo_{i}'] = response

    # Save updated DataFrame back to CSV
    try:
        output_file_path = file_path.replace('.csv', '_updated.csv')
        df.to_csv(output_file_path, index=False)
        print(f"Updated file saved: {output_file_path}")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")

    print(f"Completed processing file: {file_path}")

def process_all_files_in_directory(folder_path, model_name, hf_token):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".csv"):
            qwen_output_single_file(file_path, model_name, hf_token)

# Example usage
process_all_files_in_directory(
    folder_path="/mounts/data/proj/molly/LLM_bias_analysis/data/qwen_2.5_1.5B/system_prompt1_user_prompt_1_preference_system_prompt1_preference_user_prompt1",
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    hf_token="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"
)