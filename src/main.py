import langchain
from langchain.llms.bedrock import Bedrock
from langchain.llms import Anthropic
from anthropic import AnthropicBedrock
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
import chromadb
import torch
import boto3
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from s3_utils import list_files_in_bucket, get_file_from_s3
#from pandas_utils import generate_pandas_data
from dotenv import load_dotenv

load_dotenv()

# Load tokenizer and model from Hugging Face model hub
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
bucket_name = 'csv-rag-kb'

#ChromaDB COLLECTION
client = chromadb.Client()

def main():
    
    if "files" not in st.session_state:
        st.session_state['files'] = []
    
    if "chromadb_context" not in st.session_state:
        st.session_state['chromadb_context'] = ""
        
    #generate_pandas_data()
    if "llm" not in st.session_state:
        session = boto3.Session()
        bedrock_client = session.client(service_name="bedrock-runtime", region_name='us-east-1')

        st.session_state.llm = Bedrock(
            model_id="anthropic.claude-v2",
            client=bedrock_client,
            model_kwargs={"max_tokens_to_sample": 500},
        )
        
    if "claude" not in st.session_state:        
        st.session_state.claude = Anthropic(
            model="claude-2"
        )
    
    if "anthropic_client" not in st.session_state:        
        st.session_state.anthropic_client = AnthropicBedrock()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("CSV Chatbot")
    
    if st.button("Load Data"):
        create_chromaDB()
    
    # Initialize the agent
    if "agent" not in st.session_state:
       st.session_state.agent = create_csv_agent(
           st.session_state.llm,
           st.session_state.files,
           verbose=True,
           agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    # Chat input
    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                complete_response = generate_query(user_input)
                
                response = "No encontre respuesta. Puedes ser mas especifico por favor?"
                
                response = complete_response.content[0].text
                # response = st.session_state.agent.run(query)

            message_placeholder = st.empty()
            message_placeholder.markdown(response)

        chatbot_message = {"role": "assistant", "message": response}
        st.session_state.chat_history.append(chatbot_message)
    

def generate_query(query):    
    
    # Example query
    # query = "Cuantos clientes tiene la compañía?"
    query_embedding = generate_embeddings(query)

    collection = client.get_collection("surveys_datasource")
    # Perform the query    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=2
    )

    if "cromadb_context" not in st.session_state:
        st.session_state.cromadb_context = results
    # response = st.session_state.agent.run(query)
    
    # Usa Claude para evaluar cuál respuesta es la mejor
    # Evaluar cada respuesta generando un texto que califique la respuesta
    eval_prompt = []
    for response in st.session_state.cromadb_context:
        eval_prompt.append(f"{query}: {response}")
        # eval_response = st.session_state.claude.generate(prompts=[eval_prompt])
        # evaluations.append(eval_response)

    best_response = st.session_state.anthropic_client.messages.create(
        model="anthropic.claude-v2",
        max_tokens=2000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{query}: {st.session_state.cromadb_context}"
                    }
                ]
            }
        ]
    )

    # Suponiendo que alguna lógica de evaluación determine la mejor basada en el texto generado
    # best_response = max(evaluations, key=lambda x: x.score)  # Asumiendo que puedes extraer una puntuación del texto generado

    print(best_response)
    print(best_response.content[0].text)
    return best_response
    
# Define a function to generate embeddings
def generate_embeddings(text):
    print(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Take the mean of the output tensor over the sequence dimension to get a single vector
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
def id_generator(docs_in):
    ids = []
    for i in range(len(docs_in)):
        ids.append(f"doc_{i}")
    return ids

def create_chromaDB():
    
    # Lista los archivos en el bucket
    files = list_files_in_bucket(bucket_name)
    print(files)
    documents = get_file_from_s3(bucket_name, files[2]) if files else None
    
    ids = id_generator(documents) # Unique identifiers for each document
    #print(ids)
    embeddings = [generate_embeddings(doc) for doc in documents]
    
    collection = client.create_collection("surveys_datasource")
    
    # Add documents with embeddings to ChromaDB
    collection.add(documents=documents, embeddings=[emb.tolist() for emb in embeddings], ids=ids)

if __name__ == "__main__":
    main()
