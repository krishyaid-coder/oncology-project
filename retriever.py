from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient

embeddings = SentenceTransformerEmbeddings(model_name = 'NeuML/pubmedbert-base-embeddings')

url = 'http://localhost:6333/dashboard'

client = QdrantClient(
    url = url,
    prefer_grpc = False
)
print('This is my Qdrant client ', client)

db = Qdrant(client = client, embeddings=embeddings, collection_name='vector_database')

print('this is my db,', db)

query = 'what are side effects of systemic therapeutic agents?'

docs = db.similarity_search_with_score(query=query, k=2)


for i in docs:
    doc, score = i
    print('Document : ', doc.page_content)
    print('Score : ',score)