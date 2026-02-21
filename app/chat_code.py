import os
import typer
from typing import Annotated, Optional
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = typer.Typer(help="Fortress RAG: Chat with your codebase securely.")
console = Console()

DB_DIR = "./chroma_db"

os.environ["GOOGLE_API_KEY"] = "AIzaSyDm0AVqjPEWRgz4lhbJefQpG3GEAGPdGsI"

def get_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
    system_prompt = (
    "You are a friendly, helpful Senior Software Engineer and Research Assistant. "
    "Your personality is professional, warm, and human-like.\n\n"
    
    "HOW TO BEHAVE:\n"
    "1. **Small Talk & Etiquette**: Respond naturally to greetings, thanks, and goodbyes. "
    "If the user says 'Thanks bye', say something like 'You're very welcome! Feel free to reach out if you need more help with the code. Happy coding!'\n"
    "2. **General Coding Knowledge**: If the user asks for a common code snippet (like 'print hello world') that is NOT related to the local files, go ahead and provide it using your general knowledge, but keep it brief.\n"
    "3. **Code Research (The Grounding Rule)**: When asked about specific logic, functions, or data in the 'test-dir', use ONLY the provided context. If it's not there, don't just say 'I can't find it.' Instead, say: 'I took a look through the files, but I couldn't find a mention of [X] in this directory. Is it possible it's under a different name?'\n"
    "4. **No Hallucination**: Never invent features about the local codebase that don't exist.\n"
    "5. **Mask Secrets**: If you see an API key in the context, mask it (e.g., 'AIza...****').\n\n"
    
    "LOCAL CODE CONTEXT:\n"
    "{context}\n\n"
    "Ready to help! What's on your mind regarding the code?"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

@app.command()
def index(
    path: Annotated[Path, typer.Argument(help="The directory to scan for code")],
    reset: Annotated[bool, typer.Option("--reset", help="Delete existing index before scanning")] = False
):
    """
    [bold green]Index[/bold green] a directory into the vector store.
    """
    if reset and os.path.exists(DB_DIR):
        console.print(f"[yellow]Resetting database at {DB_DIR}...[/yellow]")
        shutil.rmtree(DB_DIR)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Reading files...", total=None)
        loader = DirectoryLoader(str(path), glob="**/*.*", loader_cls=TextLoader, silent_errors=True)
        docs = loader.load()
        
        progress.add_task(description="Splitting code into chunks...", total=None)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        progress.add_task(description="Generating embeddings (Gemini)...", total=None)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)

    console.print(f"[bold green] Indexed {len(docs)} files successfully![/bold green]")

@app.command()
def chat():
    """
    [bold cyan]Chat[/bold cyan] with your codebase using Gemini 3 Flash.
    """
    if not os.path.exists(DB_DIR):
        console.print("[red]Error: No index found. Run 'index' command first.[/red]")
        raise typer.Exit()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
    system_prompt = (
        "You are a strict Code Researcher. Answer ONLY using the provided context.\n"
        "If unsure, state you cannot find it. Mask secrets.\n\n"
        "CONTEXT:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    chain = create_retrieval_chain(
        vectorstore.as_retriever(), 
        create_stuff_documents_chain(llm, prompt)
    )

    console.print("[bold cyan] Fortress Chat Active. Type 'exit' to quit.[/bold cyan]")
    
    while True:
        query = typer.prompt("User")
        if query.lower() in ["exit", "quit"]:
            break
        
        with console.status("[bold green]Thinking..."):
            response = chain.invoke({"input": query})
            
        console.print(f"\n[bold blue]Agent:[/bold blue] {response['answer']}\n")

if __name__ == "__main__":
    app()