import os
import pandas as pd
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

def process_qwen_preference(directory_path, model_name, hf_token):
    """
    Iterates over all CSV files in a directory, compares AI-generated articles, and assigns preferences based on AI evaluation.

    Parameters:
    directory_path (str): Path to the folder containing CSV files.
    model_name (str): The Hugging Face model name (e.g., "Qwen/Qwen2.5-1.5B-Instruct").
    hf_token (str): The Hugging Face API token for model access.

    Returns:
    None (Saves updated datasets in the same directory)
    """

    # Load Qwen model and tokenizer once to avoid reloading for each file
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda:5",
        cache_dir="/mounts/data/proj/molly/cache",
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # Define valid number sets for each preference column
    valid_number_sets = {
        1: ["1", "2"],
        2: ["1", "2", "3"],
        3: ["1", "2"],
        4: ["1", "2", "3"],
        5: ["4", "5"],
        6: ["3", "4", "5"],
        7: ["4", "5"],
        8: ["3", "4", "5"],
    }

    # Define preference prompts
    preference_prompts = {
        1: "You will read two full-length articles. Based on clarity, factual accuracy, and coherence, strictly respond with '1' or '2'. The best response is:",
        2: "You will read three full-length articles. Based on clarity, factual accuracy, and coherence, strictly respond with '1', '2', or '3'. The best response is:",
        3: "Which of these two responses is less biased? Strictly respond with '1' or '2'. The best response is:",
        4: "Which of these three responses is less biased? Strictly respond with '1', '2', or '3'. The best response is:",
        5: "You will read two full-length articles. Based on clarity, factual accuracy, and coherence, strictly respond with '4' or '5'. The best response is:",
        6: "You will read three full-length articles. Based on clarity, factual accuracy, and coherence, strictly respond with '3', '4', or '5'. The best response is:",
        7: "Which of these two responses is less biased? Strictly respond with '4' or '5'. The best response is:",
        8: "Which of these three responses is less biased? Strictly respond with '3', '4', or '5'. The best response is:",
    }

    # Define prompt response mappings
    prompt_columns = ["prompt_combo_1", "prompt_combo_2", "prompt_combo_3", "prompt_combo_4", "prompt_combo_5"]

    # Iterate over all CSV files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing: {file_path}")

            # Load the dataset
            df = pd.read_csv(file_path)

            # Generate preference columns
            for i in range(1, 9):
                pref_col = f"preference_combo_{i}"
                df[pref_col] = None  # Initialize column

                # Determine which responses to compare
                if i in [1, 3]:  
                    choices = ["prompt_combo_1", "prompt_combo_2"]
                elif i in [2, 4]:  
                    choices = ["prompt_combo_1", "prompt_combo_2", "prompt_combo_3"]
                elif i in [5, 7]:  
                    choices = ["prompt_combo_4", "prompt_combo_5"]
                elif i in [6, 8]:  
                    choices = ["prompt_combo_3", "prompt_combo_4", "prompt_combo_5"]

                # Process each row using AI model
                for idx, row in df.iterrows():
                    # Create mapping of numbers to actual response texts
                    original_mapping = {choice: row[choice] for choice in choices if pd.notna(row[choice])}

                    # Shuffle the numbers and create a mapping
                    shuffled_numbers = random.sample(valid_number_sets[i], len(original_mapping))
                    shuffled_mapping = dict(zip(shuffled_numbers, original_mapping.values()))

                    # Construct chat messages with shuffled response numbers
                    messages = [
                        {"role": "system", "content": "You are a helpful AI assistant"},
                        {"role": "user", "content": preference_prompts[i] + "\n\n" + "\n\n".join([f"{num}: {text}" for num, text in shuffled_mapping.items()])}
                    ]

                    def generate_response():
                        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                        generated_ids = model.generate(
                            **model_inputs,
                            max_new_tokens=10,  
                            do_sample=False  # 🔥 Greedy decoding - always chooses the most probable token
                        )
                        return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

                    # Generate the AI response using greedy decoding
                    response = generate_response()
                    extracted_preference = next((word for word in response.split() if word in valid_number_sets[i]), None)

                    # If AI fails to pick a valid number, force a retry with stricter constraints
                    if extracted_preference is None:
                        print(f"⚠️ AI failed to provide a valid response for row {idx}. Retrying with a stricter prompt...")
                        messages[0]["content"] = "You MUST select a response. Choose from " + ", ".join(valid_number_sets[i]) + " only."
                        response = generate_response()
                        extracted_preference = next((word for word in response.split() if word in valid_number_sets[i]), None)

                    # If AI still fails, default to the lowest valid choice
                    if extracted_preference is None:
                        extracted_preference = min(valid_number_sets[i])  # Pick the lowest valid number
                        print(f"⚠️ AI failed twice. Assigning default selection '{extracted_preference}'.")

                    # Map back to the original response
                    df.at[idx, pref_col] = [key for key, value in shuffled_mapping.items() if value == shuffled_mapping[extracted_preference]][0]

            # Generate output file path
            output_file_path = file_path.replace(".csv", "_updated_Qwen_greedy.csv")
            df.to_csv(output_file_path, index=False)

            print(f"✅ Updated dataset saved as: {output_file_path}")

# Example usage
process_qwen_preference(
    directory_path="/mounts/data/proj/molly/LLM_bias_analysis/data/qwen_2.5_1.5B/system_prompt1_user_prompt_1_preference_system_prompt1_preference_user_prompt1",
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    hf_token="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"
)