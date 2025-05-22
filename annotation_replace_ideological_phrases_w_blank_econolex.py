
import os
import pandas as pd

# Root directory containing annotation_ready_articles.xlsx files
ANNOTATION_ROOT = "/mounts/data/proj/molly/LLM_bias_analysis/annotation_files_econolex"

# List of ideological phrases to blank
PHRASES_TO_REPLACE = [
    "Socialist", "socialist",
    "Capitalist", "capitalist",
    "Social", "social",
    "Socialist's", "socialist’s",  # note: both straight and curly apostrophes
    "Capitalist's", "capitalist's",
    "Socialism", "socialism",
    "Capitalism", "capitalism"
]
  
# Extra markers to normalize
MARKERS_TO_REPLACE = [
    "<blank>",
    "&lt;blank&gt;",
    "(blank)"
]

def blank_phrases(text):
    text = str(text)
    # Normalize any previous placeholder variations
    for marker in MARKERS_TO_REPLACE:
        text = text.replace(marker, "BLANK")
    # Replace ideological phrases
    for phrase in PHRASES_TO_REPLACE:
        text = text.replace(phrase, "BLANK")
    return text

def clean_annotation_ready_articles():
    for dirpath, _, filenames in os.walk(ANNOTATION_ROOT):
        for filename in filenames:
            if filename == "annotation_ready_articles.xlsx":
                file_path = os.path.join(dirpath, filename)
                print(f"\n🔍 Cleaning file: {file_path}")
                try:
                    df = pd.read_excel(file_path, engine="openpyxl")
                except Exception as e:
                    print(f"❌ Error reading {file_path}: {e}")
                    continue

                # Normalize column names for matching
                df.columns = df.columns.str.strip().str.lower()

                for col in ["article 1", "article 2", "article 3"]:
                    if col in df.columns:
                        print(f"✏️ Replacing phrases in: {col}")
                        df[col] = df[col].apply(blank_phrases)
                    else:
                        print(f"⚠️ Column '{col}' not found in: {file_path}")

                df.to_excel(file_path, index=False)
                print(f"✅ Saved cleaned file: {file_path}")

if __name__ == "__main__":
    clean_annotation_ready_articles()
