from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name = 'neuml/pubmedbert-base-embeddings')

print(embeddings)

# we are using the loader to load the data from the source i.e the Data directory.
loader = DirectoryLoader('Data/', glob='**/*.pdf', show_progress = True, loader_cls = PyPDFLoader)
documents = loader.load()

# now we do chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
texts = text_splitter.split_documents(documents)

url = 'http://localhost:6333/dashboard'

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url = url,
    prefer_grpc = False, # remote protocol
    collection_name = 'vector_database'
)
print('Vector Database created Successfully')