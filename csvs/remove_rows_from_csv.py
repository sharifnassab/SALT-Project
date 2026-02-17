import pandas as pd
from pathlib import Path

INPUT_CSV         = "results_.csv"
OUTPUT_CSV        = "results_.csv"
FIRST_COL_SUBSTR  = "LMS_MDN^HIDBD"
THIRD_COL_SUBSTR  = ''   # <-- one set of quotes
HAS_HEADER        = True
ENCODING          = "utf-8"

header = "infer" if HAS_HEADER else None
df = pd.read_csv(INPUT_CSV, header=header, dtype=str, na_filter=False, encoding=ENCODING)

if df.empty:
    df.to_csv(OUTPUT_CSV, index=False, header=(header is not None), encoding=ENCODING)
else:
    first_col_name = df.columns[0] if HAS_HEADER else 0
    third_col_name = df.columns[2] if HAS_HEADER else 2

    # literal substring match (not regex)
    to_remove_mask1 = df[first_col_name].astype(str).str.contains(FIRST_COL_SUBSTR, na=False, regex=False)
    to_remove_mask3 = df[third_col_name].astype(str).str.contains(THIRD_COL_SUBSTR, na=False, regex=False)

    to_remove_mask = to_remove_mask1 & to_remove_mask3
    rows_to_drop = df.index[to_remove_mask].tolist()

    print(f"Matches in first column:  {to_remove_mask1.sum()}")
    print(f"Matches in third column:  {to_remove_mask3.sum()}")
    print(f"Rows meeting BOTH (drop): {to_remove_mask.sum()}")

    df_filtered = df.drop(index=rows_to_drop).reset_index(drop=True)
    df_filtered.to_csv(OUTPUT_CSV, index=False, header=(header is not None), encoding=ENCODING)

print(f"Done. Wrote: {Path(OUTPUT_CSV).resolve()}")
