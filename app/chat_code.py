import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# Set your API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDm0AVqjPEWRgz4lhbJefQpG3GEAGPdGsI"

def main():
    if len(sys.argv) < 2:
        print("Usage: python chat_code.py <directory_path>")
        return

    target_dir = sys.argv[1]
    
    print(f"🔍 Indexing {target_dir}...")
    
    # 1. Load files (simple TextLoader for maximum compatibility)
    loader = DirectoryLoader(target_dir, glob="**/*.*", loader_cls=TextLoader, 
                             silent_errors=True, recursive=True)
    docs = loader.load()
    
    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 3. Create Vector Store (In-memory for speed, or persist to disk)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # 4. Setup Gemini 3 Flash
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

    print("✅ Ready! Type 'exit' to quit.")
    
    while True:
        query = input("\n💬 Query: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        response = qa_chain.invoke(query)
        print(f"\n🤖: {response['result']}")

if __name__ == "__main__":
    main()