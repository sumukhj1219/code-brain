import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 2026 Correct Imports for Retrieval Chains
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# SECURE YOUR KEY: Masking it here for the example, but it will use your env var
os.environ["GOOGLE_API_KEY"] = "AIzaSyDm0AVqjPEWRgz4lhbJefQpG3GEAGPdGsI"

def get_fortress_chain(vectorstore, llm):
    """Encapsulates the strict system prompt logic."""
    system_prompt = (
        "You are a strict Code Research Assistant. Your ONLY source of truth is the provided context from the 'test-dir' directory.\n\n"
        "RULES OF ENGAGEMENT:\n"
        "1. SCOPE: Answer ONLY using the provided context. If the answer is not in the context, say: 'I cannot find information about this in the local directory.'\n"
        "2. NO HALLUCINATION: Never invent functions, variables, or logic that do not exist in the snippets.\n"
        "3. NO FILE MODIFICATION: You are a read-only researcher. Never suggest or output commands to overwrite, delete, or change files.\n"
        "4. SECRETS & PRIVACY: If you detect API keys or secrets in the code, MASK THEM (e.g., 'sk-****') and warn the user.\n"
        "5. CODE EXPLANATION ONLY: Focus on explaining how the code works and its relationships.\n"
        "6. CITATION: Mention the filepath the information came from.\n\n"
        "CONTEXT FROM LOCAL FILES:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create the modern chain structure
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

def main():
    if len(sys.argv) < 2:
        print("Usage: python chat_code.py <directory_path>")
        return

    target_dir = sys.argv[1]
    print(f"🔍 Indexing {target_dir}...")
    
    # Load and Split
    loader = DirectoryLoader(target_dir, glob="**/*.*", loader_cls=TextLoader, silent_errors=True, recursive=True)
    docs = loader.load()
    
    if not docs:
        print("❌ No documents found. Check your directory path.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Models
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
    
    # Initialize the specific RAG chain with your prompt
    rag_chain = get_fortress_chain(vectorstore, llm)

    print("✅ Fortress Mode Active! Type 'exit' to quit.")
    
    while True:
        query = input("\n💬 Query: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        # Invoke the new chain (returns a dict with 'context' and 'answer')
        response = rag_chain.invoke({"input": query})
        print(f"\n🤖 Agent: {response['answer']}")

if __name__ == "__main__":
    main()