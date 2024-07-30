import pandas as pd
from io import StringIO

# Sample CSV data
file_path = 'hey.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Sort by vy and then by vx
df_sorted = df.sort_values(by=['vy', 'vx'])
df_sorted[['vx', 'vy']] = df_sorted[['vy', 'vx']]

output_file_path = 'sorted_swapped_file.csv'  # Replace with your desired output file path
df_sorted.to_csv(output_file_path, index=False)