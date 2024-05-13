import boto3
import torch
import os
from transformers import AutoTokenizer, AutoModel
from langchain_text_splitters import CharacterTextSplitter
import requests

# Load tokenizer and model from Hugging Face model hub
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
def create_s3_client():
    """Crea un cliente de S3."""
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    return boto3.client('s3',
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,)
    #return boto3.client('s3')

def list_files_in_bucket(bucket_name):
    """Lista los archivos en un bucket especificado."""
    s3 = create_s3_client()
    response = s3.list_objects_v2(Bucket=bucket_name)
    return [item['Key'] for item in response.get('Contents', [])]

def get_file_from_s3(bucket_name, key):
    """Descarga un archivo desde S3 y lo retorna como string."""
    s3 = create_s3_client()
    response = s3.get_object(Bucket=bucket_name, Key=key)
    object_data = response['Body'] 
    chunk_size = 41280 # Specify the desired chunk size (in bytes) 
    chunks = []
    while True: 
        chunk = object_data.read(chunk_size) 
        if not chunk: 
            break 
        # Process the chunk of data as needed 
        # Example: print the chunk 
        chunk = chunk.decode('iso-8859-1', errors='replace')
        chunks.append(chunk)
        #get_embedding(chunk)
        
    return chunks

def get_embedding(text):
    text = text.decode('iso-8859-1', errors='replace')
    print(text)
    generate_embeddings(text)
    #url = "https://search-rag-domain-2-otipijwz2jpodmo6ruwx6psvu4.us-east-1.es.amazonaws.com/embed"
    #payload = {"text": text}
    #response = requests.get(url, json=payload)
    #if response.status_code == 200:
    #    return response.json()['embedding']
    #else:
    #    print("Error:", response.status_code, response.text)
    #    raise Exception(response)

    # return response['Body'].read().decode('utf-8')
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the mean of the output tensor over the sequence dimension to get a single vector
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    