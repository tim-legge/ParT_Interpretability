import pyarrow.parquet as pq

# Replace 'your_file.parquet' with the path to your Parquet file
parquet_file = pq.ParquetFile('/mnt/c/Interpretability_Data/QuarkGluon/test_file_0.parquet')

# Print the schema
print("Schema:")
print(parquet_file.schema)

# Alternatively, to get a more detailed representation of the schema
#print("\nDetailed schema:")
#print(parquet_file.schema.to_string())