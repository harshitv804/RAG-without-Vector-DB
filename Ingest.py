from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

loader = DirectoryLoader('data', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

file = open('data.pkl', 'wb')
pickle.dump(texts, file)
file.close()
