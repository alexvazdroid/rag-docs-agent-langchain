import boto3
import pandas as pd
from s3_utils import list_files_in_bucket

bucket_name = 'csv-rag-kb'
files = list_files_in_bucket(bucket_name)

def generate_pandas_data():
    df = pd.read_csv(files[2])
    print(df)
    return df