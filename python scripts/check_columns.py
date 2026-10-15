"""
check_columns.py
------------------------------------------------------------------------------
One-off helper: prints the column names in Step 1's master output file so we
can confirm the exact sensor/temperature column names before running
extra01_combined_slope_lines.py.
------------------------------------------------------------------------------
"""

import pandas as pd

MASTER_PATH = (
    r"C:\Users\c2051290\OneDrive - Newcastle University\NCLU\PhD\My PhD"
    r"\01 Onsite Work and Engineering\01 Sensors\02 Canberra analysis"
    r"\06 Files for Python Final Round\Output\Round 2"
    r"\Step 1 Collation Alignment and Raw Plots\All_Sensors_Aligned_Filled_Trimmed.csv"
)

df = pd.read_csv(MASTER_PATH, nrows=0)
print("Columns found in master file:")
for col in df.columns:
    print(f"  - {col!r}")
