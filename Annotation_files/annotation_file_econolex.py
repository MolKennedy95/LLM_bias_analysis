import os
import pandas as pd
import random

# Input and output directories
ROOT_DIR = "/mounts/data/proj/molly/LLM_bias_analysis/EconoLex_data"
OUTPUT_ROOT = "/mounts/data/proj/molly/LLM_bias_analysis/annotation_files_econolex"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

label_map = {
    1: "Socialist",
    2: "Capitalist",
    3: "Neutral"
}

def process_topic_file(model_name, title_file):
    input_path = os.path.join(ROOT_DIR, model_name, title_file)

    try:
        df = pd.read_excel(input_path, engine="openpyxl")
    except Exception as e:
        print(f"❌ Error reading {input_path}: {e}")
        return

    if "Title" not in df.columns:
        print(f"⚠️ Skipping {input_path}: 'Title' column not found.")
        return

    # Filter rows that have all 3 prompt combos
    valid_rows = []
    for _, row in df.iterrows():
        if pd.isna(row["Title"]):
            continue
        has_all = all(
            f"prompt_combo_{i}" in df.columns and pd.notna(row[f"prompt_combo_{i}"]) for i in range(1, 4)
        )
        if has_all:
            valid_rows.append(row)

    # Select up to 10 unique titles
    selected_rows = []
    seen_titles = set()
    for row in valid_rows:
        title = row["Title"]
        if title not in seen_titles:
            selected_rows.append(row)
            seen_titles.add(title)
        if len(seen_titles) == 1000:
            break

    if len(selected_rows) < 1000:
        print(f"⚠️ Not enough complete rows in {input_path}")
        return

    # Prepare output
    annot_rows = []
    key_rows = []

    for row in selected_rows:
        topic = row["Title"]

        articles = [
            {"text": str(row["prompt_combo_1"]).strip(), "label": "Socialist"},
            {"text": str(row["prompt_combo_2"]).strip(), "label": "Capitalist"},
            {"text": str(row["prompt_combo_3"]).strip(), "label": "Neutral"},
        ]
        random.shuffle(articles)

        annot_rows.append({
            "Title": topic,
            "Article 1": articles[0]["text"],
            "Article 2": articles[1]["text"],
            "Article 3": articles[2]["text"],
            "Your Guess 1": "",
            "Your Guess 2": "",
            "Your Guess 3": ""
        })

        key_rows.append({
            "Title": title,
            "Shuffled Order": f"[{articles[0]['label']}, {articles[1]['label']}, {articles[2]['label']}]"
        })

    title_name = os.path.splitext(title_file)[0]
    output_dir = os.path.join(OUTPUT_ROOT, model_name, title_name)
    os.makedirs(output_dir, exist_ok=True)

    df_annot = pd.DataFrame(annot_rows)
    df_key = pd.DataFrame(key_rows)

    df_annot.to_excel(os.path.join(output_dir, "annotation_ready_articles.xlsx"), index=False)
    df_key.to_excel(os.path.join(output_dir, "article_shuffled_key.xlsx"), index=False)

    print(f"✅ Saved outputs for {model_name}/{title_file} → {output_dir}")

def run_all_models():
    model_dirs = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    for model_name in model_dirs:
        model_path = os.path.join(ROOT_DIR, model_name)
        title_files = [f for f in os.listdir(model_path) if f.endswith(".xlsx")]
        for title_file in title_files:
            process_topic_file(model_name, title_file)

if __name__ == "__main__":
    run_all_models()

