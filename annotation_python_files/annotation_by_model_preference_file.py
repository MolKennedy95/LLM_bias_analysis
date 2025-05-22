import pandas as pd
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse, os, random, re

# Suppress unnecessary logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Utility function to extract a number from the model's response
def extract_number(text):
    match = re.search(r'[-+]?(?:\d*\.?\d+)', text)
    if match:
        return str(int(match.group()))
    return None

def find_preferred_article(folder_path, model_name, device, hf_token):
    print(f"🔍 Processing folder: {folder_path}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device,
        token=hf_token,
        cache_dir=,
        trust_remote_code=True
    )

    valid_choices = ["1", "2", "3"]
    guess_cols = ['your guess 1', 'your guess 2', 'your guess 3']

    for filename in os.listdir(folder_path):
        if not filename.endswith(".xlsx"):
            continue

        file_path = os.path.join(folder_path, filename)
        print(f"\n📁 Processing file: {file_path}")

        try:
            df = pd.read_excel(file_path, engine="openpyxl")
        except Exception as e:
            print(f"❌ Failed to read {file_path}: {e}")
            continue

        # Ensure expected guess columns exist
        for col in guess_cols:
            if col not in df.columns:
                raise ValueError(f"Missing expected column: '{col}' in {file_path}")

        for idx, row in df.iterrows():
            articles = []
            for i, col in enumerate(['article 1', 'article 2', 'article 3'], 1):
                if pd.notna(row[col]):
                    articles.append((str(i), row[col]))

            if len(articles) < 3:
                print(f"⚠️ Skipping row {idx} due to missing article data.")
                continue

            # Construct prompt for the LLM
            prompt_intro = (
                "You will read three articles. The three articles are on the same topic but written from three different perspectives. "
                "A neutral article emphasizes on: Balanced, fact-based reporting, Absence of emotional or ideological language,Equal presentation of viewpoints and reliance on data"
                "Which article do you think is written by the most neutral author?"
                "Respond strictly with '1', '2', or '3'.\n\n"
            )
            prompt_articles = "\n\n".join([f"Article {num}:\n{text}" for num, text in articles])
            prompt = prompt_intro + prompt_articles

            # Prepare message in chat format
            messages = [{"role": "user", "content": prompt}]
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            chat_text += "\n\nThe most neutral article is:"
            # print(chat_text)


            def generate_response():
                inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=6,
                    do_sample=False  # Greedy decoding
                )
                return tokenizer.decode(output_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True).strip()

            response = generate_response()
            preferred_choice = extract_number(response)
            print("------------------------------------------------------------------------------------------------------")


            print(f"🧠 Row {idx} → Model Response: '{response}' → Parsed: {preferred_choice}")

            if preferred_choice not in valid_choices:
                print(f"⚠️ Invalid response at row {idx}. Setting all guesses to -1.")
                for col in guess_cols:
                    df.at[idx, col] = -1
            else:
                choice_idx = int(preferred_choice) - 1
                df.at[idx, guess_cols[choice_idx]] = "neutral"

        # Save to a new file instead of overwriting
        try:
            new_filename = "model_preferred_article.xlsx"
            new_path = os.path.join(folder_path, new_filename)

            df.to_excel(new_path, index=False, engine="openpyxl")
            print(f"✅ Saved annotated file as: {new_path}")
        except Exception as e:
            print(f"❌ Error saving {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help=)
    parser.add_argument("--folder_path", type=str, required=True, help="Path to folder containing Excel files")
    parser.add_argument("--device", type=str, default="", help="Device to run model on (e.g., 'cuda:0' or 'cpu')")
    args = parser.parse_args()

    find_preferred_article(
        folder_path=args.folder_path,
        model_name=args.model_name,
        device=args.device,
        hf_=
    )

if __name__ == "__main__":
    main()
