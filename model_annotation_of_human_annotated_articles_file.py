import os
import re
import pandas as pd
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

# Suppress logging from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

def classify_most_neutral(file_path, model_path, device):
    # Load tokenizer and model from local cache
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device,
        trust_remote_code=True,
        cache_dir="/mounts/data/proj/molly/cache",
        token=,
    )

    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return

    # Ensure "Model" column exists
    df["Model"] = None

    for idx, row in df.iterrows():
        articles = [row.get(f'article {i}', '') for i in range(1, 4)]

        prompt = (
            "You will be shown three articles. Identify the one that is most neutral in tone. "
            "A neutral article emphasizes on: Balanced, fact-based reporting, Absence of emotional or ideological language,Equal presentation of viewpoints and reliance on data"
            "Which article do you think is written by the most neutral author?"
            "Respond only with the number of the most neutral article: 1, 2, or 3.\n\n"
        )

        try:
            
            for i, article in enumerate(articles, 1):
                prompt += f"Article {i}:\n{article.strip()}\n\n"

        except: 
            continue


        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text += "\n\nThe most neutral article is:"

        def generate_response():
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            return tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True).strip()

        try:
            response = generate_response()
            print(f"[Row {idx}] Model response: {response}")
            match = re.search(r'\b([1-3])\b', response)
            if match:
                guess = match.group(1)
                df.at[idx, "Model"] = f"a{guess}"
            else:
                df.at[idx, "Model"] = "unknown"
        except Exception as e:
            print(f"⚠️ Error on row {idx}: {e}")
            df.at[idx, "Model"] = "error"

    # Save the updated DataFrame
    output_path = file_path.replace(".xlsx", f"with{model_path.split('/')[-1]}.xlsx")
    try:
        df.to_excel(output_path, index=False, engine="openpyxl")
        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save file: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to Excel file with articles")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the local cached model directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load model on (e.g., 'cuda:0')")
    args = parser.parse_args()

    classify_most_neutral(
        file_path=args.file_path,
        model_path=args.model_name,
        device=args.device
    )

if __name__ == "__main__":
    main()
