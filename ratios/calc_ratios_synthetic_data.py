import pandas as pd
import os

def process_files(folder_path, ratios_base_dir):
    # Create a central 'ratios' directory
    os.makedirs(ratios_base_dir, exist_ok=True)
    
    # Define a subdirectory inside 'ratios' for each input directory
    ratios_dir = os.path.join(ratios_base_dir, os.path.basename(folder_path))
    os.makedirs(ratios_dir, exist_ok=True)
    
    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx"):  # Ensure only Excel files are processed
            file_path = os.path.join(folder_path, filename)
            df = pd.read_excel(file_path)
            
            # Dictionary to store ratios for each column
            ratios = {}

            # Calculate ratio for each specified column
            for column in [
                "preference_combo_1", "preference_combo_2", "preference_combo_3", 
                "preference_combo_4", "preference_combo_5", "preference_combo_6", 
                "preference_combo_7", "preference_combo_8"
            ]:
                if column in df.columns:  # Ensure the column exists in the file
                    column_ratios = df[column].value_counts(normalize=True).to_dict()
                    ratios[column] = column_ratios
                else:
                    print(f"Column {column} not found in {filename}")

            # Convert the ratios dictionary to a DataFrame for saving
            ratios_df = pd.DataFrame(ratios).fillna(0)  # Fill missing values with 0

            # Save the ratios to the central 'ratios' directory
            output_file = os.path.join(ratios_dir, f"synthetic_macro_avg_{filename}")
            ratios_df.to_excel(output_file, index_label="Value")

            print(f"Processed {filename}, ratios saved to {output_file}")

    print(f"Processing complete for {folder_path}")

if __name__ == "__main__":
    # List of directories containing the Excel files
    directories = ["/mounts/data/proj/molly/LLM_bias_analysis/data/c4ai-command-r7b-12-2024", "/mounts/data/proj/molly/LLM_bias_analysis/data/DeepSeek-R1-Distill-Llama-8B", "/mounts/data/proj/molly/LLM_bias_analysis/data/gemma-2-9b-it",
                   "/mounts/data/proj/molly/LLM_bias_analysis/data/GLM-4-9B-Chat", "/mounts/data/proj/molly/LLM_bias_analysis/data/llama_3.1_8B",
                   "/mounts/data/proj/molly/LLM_bias_analysis/data/Ministral-8B-Instruct-2410", "/mounts/data/proj/molly/LLM_bias_analysis/data/phi-4",
                   "/mounts/data/proj/molly/LLM_bias_analysis/data/Qwen2.5-7B-Instruct"]
    
    # Define the central ratios directory
    ratios_base_dir = "/mounts/data/proj/molly/LLM_bias_analysis/macro_avg"
    
    for directory in directories:
        process_files(directory, ratios_base_dir)
