import pandas as pd
import os

def process_files(folder_path, ratios_base_dir):
    # Create a central 'ratios' directory
    os.makedirs(ratios_base_dir, exist_ok=True)
    
    # Define a subdirectory inside 'ratios' for each input directory
    ratios_dir = os.path.join(ratios_base_dir, os.path.basename(folder_path))
    os.makedirs(ratios_dir, exist_ok=True)
    
    ratios = {}
    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx"):  # Ensure only Excel files are processed
            file_path = os.path.join(folder_path, filename)
            df = pd.read_excel(file_path)
            
            # Dictionary to store ratios for each column

            # Calculate ratio for each specified column
            for column in [
                "preference_combo_1", "preference_combo_2", "preference_combo_3", 
                "preference_combo_4", "preference_combo_5", "preference_combo_6", 
                "preference_combo_7", "preference_combo_8"
            ]:
                if column in df.columns:  # Ensure the column exists in the file
                    column_ratios = df[column].value_counts().to_dict()
                    if column not in ratios:
                        ratios[column] = column_ratios
                        # print("here", column_ratios)
                    else:
                        for ratio_key in column_ratios:
                            # print('ratio_key', ratio_key)
                            if ratio_key not in ratios[column]:
                                ratios[column][ratio_key] = column_ratios[ratio_key]
                            else:
                                ratios[column][ratio_key] += column_ratios[ratio_key]
                else:
                    print(f"Column {column} not found in {filename}")
            print(f"Processed {filename}")

            # Convert the ratios dictionary to a DataFrame for saving
    ratios_df = pd.DataFrame(ratios).fillna(0)  # Fill missing values with 0
    ratios_df = ratios_df.div(ratios_df.sum())
    # Save the ratios to the central 'ratios' directory
    output_file = os.path.join(ratios_dir, f"synthetic_micro_avg.xlsx")
    ratios_df.to_excel(output_file, index_label="Value")


    print(f"Processing complete for {folder_path}")

if __name__ == "__main__":
    # List of directories containing the Excel files
    directories = ["/mounts/data/proj/molly/LLM_bias_analysis/data/c4ai-command-r7b-12-2024", "/mounts/data/proj/molly/LLM_bias_analysis/data/DeepSeek-R1-Distill-Llama-8B", "/mounts/data/proj/molly/LLM_bias_analysis/data/gemma-2-9b-it",
                   "/mounts/data/proj/molly/LLM_bias_analysis/data/GLM-4-9B-Chat", "/mounts/data/proj/molly/LLM_bias_analysis/data/llama_3.1_8B",
                   "/mounts/data/proj/molly/LLM_bias_analysis/data/Ministral-8B-Instruct-2410", "/mounts/data/proj/molly/LLM_bias_analysis/data/phi-4",
                   "/mounts/data/proj/molly/LLM_bias_analysis/data/Qwen2.5-7B-Instruct"]
    
    # Define the central ratios directory
    ratios_base_dir = "/mounts/data/proj/molly/LLM_bias_analysis/micro_averaged_ratios"
    
    for directory in directories:
        process_files(directory, ratios_base_dir)
