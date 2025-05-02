import pandas as pd
import shutil

def duplicate_csv_file(original_file, new_file):
    try:
        shutil.copyfile(original_file, new_file)
        print(f"File copied successfully: '{original_file}' → '{new_file}'")
    except Exception as e:
        print(f"Error copying file: {e}")

def get_exact_match_indices(csv_file, col_idx, var_match):
    try:
        df = pd.read_csv(csv_file, header=None, keep_default_na=False, na_values=[])
        return df.index[df.iloc[:, col_idx] == var_match].tolist() if col_idx < df.shape[1] else []
    except Exception:
        return []

def get_partial_match_indices(csv_file, col_idx, var_include):
    try:
        df = pd.read_csv(csv_file, header=None, keep_default_na=False, na_values=[])
        return df.index[df.iloc[:, col_idx].astype(str).str.contains(var_include, na=False)].tolist() if col_idx < df.shape[1] else []
    except Exception:
        return []

def sort_csv_by_columns(csv_file, col_indices, output_file=None):
    if output_file is None:
        output_file = csv_file
    try:
        df = pd.read_csv(csv_file, header=None, keep_default_na=False, na_values=[])
        cols = [df.columns[i] for i in col_indices if i < df.shape[1]]
        df.sort_values(by=cols).to_csv(output_file, index=False)
    except Exception:
        pd.read_csv(csv_file, header=None, keep_default_na=False, na_values=[]).to_csv(output_file, index=False, na_rep='nan')

def replace_in_rows(
    csv_file, row_indices,
    col_to_replace=None, replace_with=None,
    col_to_replace_substring=None, old_substring="", new_substring="",
    output_file="output.csv"
):
    df = pd.read_csv(csv_file, header=None, keep_default_na=False, na_values=[])

    if not row_indices:
        df.to_csv(output_file, index=False)
        return

    for idx in row_indices:
        if idx >= len(df):
            continue

        # Full cell replacement
        if col_to_replace is not None and col_to_replace < df.shape[1]:
            df.iat[idx, col_to_replace] = replace_with

        # Substring replacement
        if col_to_replace_substring is not None and col_to_replace_substring < df.shape[1]:
            val = df.iat[idx, col_to_replace_substring]
            if pd.notna(val):
                df.iat[idx, col_to_replace_substring] = str(val).replace(old_substring, new_substring)

    df.to_csv(output_file, index=False, na_rep='nan')

def remove_rows_by_indices(csv_file, row_indices_to_remove, output_file=None):
    if output_file is None:
        output_file = csv_file
    df = pd.read_csv(csv_file, header=None, keep_default_na=False, na_values=[])
    if not row_indices_to_remove:
        df.to_csv(output_file, index=False)
        return
    # Keep rows not in the list
    df = df.drop(index=[i for i in row_indices_to_remove if i < len(df)])
    df.to_csv(output_file, index=False, na_rep='nan')









'''
column headers:
0: Algorithm	
1: Model	
2: Parameters	
4: Dataset	
'''

if __name__=="__main__":


    source_csv_file_name = 'csvs/for_plotting5_modified.csv'
    output_csv_file_name = 'csvs/for_plotting6_.csv'

    # duplicate the file
    duplicate_csv_file(source_csv_file_name, output_csv_file_name)

        
    # Sort CSV
    if True:
        sort_csv_by_columns(output_csv_file_name,
                            col_indices=[4, 0, 2])

    # Remove rows from CSV:
    if False:
        partial_matches = get_partial_match_indices(output_csv_file_name, 2, "meta_eta")
        remove_rows_by_indices(output_csv_file_name, 
                               row_indices_to_remove=partial_matches)

    # 4. Replace content
    if False:
        #exact_matches = get_exact_match_indices(output_csv_file_name, 0, "apple")
        partial_matches = get_partial_match_indices(output_csv_file_name, 0, "LMS-MDNPN")
        #row_indices = list(set(exact_matches) & set(partial_matches))
        row_indices = partial_matches
        replace_in_rows(output_csv_file_name, 
                        row_indices=row_indices, 
                        col_to_replace=0, replace_with="LMS-MDNPN2",
                        col_to_replace_substring=None, old_substring="LMS-MDNPN", new_substring="LMS-MDNPN2"
        )

    exact_matches = get_exact_match_indices(output_csv_file_name, 0, "0")
    remove_rows_by_indices(output_csv_file_name, row_indices_to_remove=exact_matches)