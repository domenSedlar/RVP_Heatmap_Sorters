from pathlib import Path
import pandas as pd

# Define the directory containing your files
data_dir = Path("./Results/")

# Get all .parquet files in the directory (ignoring non-parquet files)
parquet_files = list(data_dir.glob("*.parquet"))

if not parquet_files:
    print("No .parquet files found in the directory.")
else:
    # Read each parquet file into a DataFrame and combine them
    df_list = [pd.read_parquet(file) for file in parquet_files]
    res_df = pd.concat(df_list, ignore_index=True)

    # Save the combined DataFrame to res.parquet
    output_path = data_dir / "res.parquet"
    res_df.to_parquet(output_path, index=False)

    print(f"Successfully merged {len(parquet_files)} files into {output_path}")