import streamlit as st
from huggingface_hub import login
from langchain_core.prompts import ChatPromptTemplate

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# model to be used
hf_token = os.getenv('HUGGINGFACE_TOKEN')
if hf_token:
    login(hf_token)
# model = "epfl-llm/meditron-7b"

# initiliase the qdrant db connection
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient

embeddings = SentenceTransformerEmbeddings(model_name = 'neuml/pubmedbert-base-embeddings')
url = 'http://localhost:6333/dashboard'

client = QdrantClient(
    url = url,
    prefer_grpc = False
)
db = Qdrant(client = client, embeddings=embeddings, collection_name='vector_database')

print('this is my db,', db)

# create a retriever object
print('create retriever object')
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print('retriever object create successfully')

# initialise a chat prompt template
print('Creating prompt template')
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
print('prompt template created successfully')
# Initialise a generator i.e model
# llm = pipeline("text-generation", model=model)
from langchain_groq import ChatGroq
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")
llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.1-70b-versatile")

# Initialise a output parser
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

# Define the RAG Chain
from langchain_core.runnables import RunnablePassthrough

rag_chain = {'context':retriever , 'question':RunnablePassthrough()} | prompt_template | llm | parser

# ask the question
#query = 'what are side effects of systemic therapeutic agents?'

#response = rag_chain.invoke(query)
#print(response)

st.title('Welcome to RAG based smart Oncologist!')

query = st.text_input("Please enter your query: ")
response = rag_chain.invoke(query)
st.write(response)
