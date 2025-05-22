

import pandas as pd
from sklearn.metrics import cohen_kappa_score
import argparse

def calc_agreement(file_path):

    # Load Excel
    df = pd.read_excel(file_path)
    df = df.dropna(how='all')
    df = df.applymap(lambda x: x.replace("_x000d_","") if isinstance(x, str) else x)
    # Define valid labels
    valid_labels = ['a1', 'a2', 'a3']

    # Filter to rows where BOTH Human and model are valid and not missing
    df_filtered = df[df['Human_neutral'].isin(valid_labels) & df['Model'].isin(valid_labels)]

    # df_filtered = df

    print(df_filtered)

    # Convert to string (in case of accidental float)
    labels_1 = df_filtered['Human_neutral'].astype(str)
    labels_2 = df_filtered['Model'].astype(str)

    # Compute Cohen’s Kappa
    kappa = cohen_kappa_score(labels_1, labels_2)

    print(f"Cohen’s Kappa: {kappa:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to Excel file with articles")
    args = parser.parse_args()
    calc_agreement(args.file_path)

if __name__ == "__main__":
    main()



