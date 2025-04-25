!pip install -U langchain langchain-community faiss-cpu pypdf sentence-transformers transformers accelerate
from google.colab import files
uploaded = files.upload()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

loader = PyPDFLoader("Insurance_Policies (1).pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

llm_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=256, temperature=0.7)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
while True:
    query = input("You: ")
    if query.lower() in ['exit', 'quit']:
        print("Bot: Goodbye!")
        break
    response = qa_chain.run(query)
    print("Bot:", response)
