import os
import pandas as pd
import argparse

def process_ratios(base_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get subdirectories
    subdirectories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d != "averaged_ratios"]
    
    for subdir in subdirectories:
        subdir_path = os.path.join(base_dir, subdir)
        
        # List all Excel files in the subdirectory
        excel_files = [f for f in os.listdir(subdir_path) if f.endswith(".xlsx")]
        
        # List to hold dataframes
        dfs = []
        
        for file in excel_files:
            file_path = os.path.join(subdir_path, file)
            df = pd.read_excel(file_path)
            dfs.append(df)
        
        # Concatenate all dataframes and compute the mean
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)

            print(f"combined dfs for {subdir}")
            print(combined_df)

            # Group by 'Value' and compute mean for preference columns
            avg_df = combined_df.groupby('Value').mean().reset_index()
            
            # Output file path
            output_file = os.path.join(output_dir, f"{subdir}_averaged_ratios.xlsx")
            
            # Save to Excel
            avg_df.to_excel(output_file, index=False)
            print(f"Averaged data saved to {output_file}")
        else:
            print(f"No Excel files found in {subdir_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and average Excel files in subdirectories.")
    parser.add_argument("--base_dir", type=str, help="Base directory containing subdirectories with Excel files.")
    parser.add_argument("--output_dir", type=str, nargs="?", default="/mounts/data/proj/molly/LLM_bias_analysis/averaged_ratios", help="Directory to save the averaged output files. Defaults to /mounts/data/proj/molly/LLM_bias_analysis/averaged_ratios.")
    
    args = parser.parse_args()
    process_ratios(args.base_dir, args.output_dir)
