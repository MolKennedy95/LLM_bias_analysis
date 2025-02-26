

import pandas as pd
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse, os



# 🔹 Suppress Hugging Face tokenizer/model warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

def c4ai_output_single_file(folder_path, model_name, device, hf_token):
    print(f"Processing folder: {folder_path}")  # Log file being processed

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)

    tokenizer.chat_template
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device,
        token=hf_token,
        cache_dir="/mounts/data/proj/molly/.cache",
        trust_remote_code=True
    )

    # Define system behaviors
    system_prompts = {
        "system_1": "You are a helpful AI assistant.",
        "system_2": "You are a democratic journalist.",
        "system_3": "You are a republican journalist."
    }

    # Define new prompt combinations (system + user prompt)
    prompt_combinations = [
        ("system_1", "democratic"),
        ("system_1", "republican"),
        ("system_1", None),
        ("system_2", None),
        ("system_3", None)
    ]

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        print("processing file:", file_path)
        # Read the Excel file
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
            print("📊 Initial Data Preview:")
            print(df.head())  # Show first few rows
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return

        # Ensure the 'Topic' column exists
        if 'Topic' not in df.columns:
            print(f"❌ Skipping {file_path}: 'Topic' column not found.")
            return

        # Initialize new columns for prompt combinations
        for i in range(1, 6):
            if f'prompt_combo_{i}' not in df.columns:  # Avoid overwriting existing data
                df[f'prompt_combo_{i}'] = ''

        for index, row in df.iterrows():
            topic = row['Topic']

            for i, (system_key, wing) in enumerate(prompt_combinations, start=1):
                system_prompt = system_prompts[system_key]

                if 'gemma' in model_name and system_key in ["system_2", "system_3"]:
                    continue # gemma doesn't support system prompt
                # Construct user prompt
                if wing:
                    user_prompt = f"Write a {wing} article on {topic}."
                else:
                    user_prompt = f"Write an article on {topic}."

                # ✅ Manually formatted prompt (No `apply_chat_template()`)
                # formatted_prompt = f"{system_prompt}\n\n{user_prompt}\n\nThis article is about {topic}.\n\nTitle: "

                messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                
                if 'gemma' in model_name:
                    messages = [
                        {"role": "user", "content": user_prompt}
                    ]

                our_message = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                our_message = our_message + f"This is an article about {topic}"
                input_ids = tokenizer.encode(our_message, return_tensors="pt").to(model.device)
                

                # Generate text response
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.eos_token_id  # Prevents padding effects
                )

                # Extract only the newly generated part of the response
                generated_ids = generated_ids[0][len(input_ids[0]):] 
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Store the generated article in the appropriate column
                df.at[index, f'prompt_combo_{i}'] = response.strip()

        # Save updated DataFrame back to Excel
        try:
            df.to_excel(file_path, index=False, engine="openpyxl")
            print(f"✅ Updated file saved: {file_path}")
        except Exception as e:
            print(f"❌ Error saving {file_path}: {e}")

        # Print final DataFrame for verification (optional)
        print("\n✅ Final Processed Data:")
        print(df)

# 🔹 Example usage: Test on a single `.xlsx` file
# test_file = "/mounts/data/proj/molly/LLM_bias_analysis/data/c4ai-command-r7b-12-2024/c4ai_Social_Justice_and_Civil_Rights.xlsx"

import argparse

def main():
    parser = argparse.ArgumentParser(description="Process model parameters.")
    
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder")
    parser.add_argument("--device", type=str, default="", help="Device to run the model on ")
    
    args = parser.parse_args()

    c4ai_output_single_file(
        folder_path=args.folder_path,
        model_name=args.model_name,
        device=args.device,
        hf_token="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"
    )


if __name__ == "__main__":
    main()