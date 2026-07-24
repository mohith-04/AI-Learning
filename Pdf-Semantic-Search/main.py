from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader("data/employee_policy.pdf")
documnets=loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks=text_splitter.split_documents(documnets)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
question = input("Ask a question: ")
results = vectorstore.similarity_search(
    question,
    k=3
)

print("\nTop Matching Chunks:\n")

for i, doc in enumerate(results, start=1):
    print(f"Chunk {i}")
    print(doc.page_content)
    print(doc.metadata)
    print()