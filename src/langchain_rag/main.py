from pathlib import Path

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = OllamaEmbeddings(model="embeddinggemma")
vector_store = InMemoryVectorStore(embeddings)
llm = ChatOllama(model="llama3.2")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def index(directory: Path) -> None:
    """Split every document in the directory into chunks and store their embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=30, keep_separator=False
    )
    for file in sorted(directory.iterdir()):
        if not file.is_file():
            continue
        vector_store.add_texts(splitter.split_text(file.read_text()))

def answer(question: str) -> str:
    """Retrieve the most relevant chunks and let the LLM answer from them."""
    relevant = vector_store.similarity_search(question, k=3)
    context = "\n\n".join(chunk.page_content for chunk in relevant)
    prompt = f"Answer using only this context:\n\n{context}\n\nQuestion: {question}"
    return llm.invoke(prompt).content

def run() -> None:
    index(PROJECT_ROOT / "data/raw")
    while True:
        question = input("\nQuestion (empty to quit): ").strip()
        if not question:
            break
        print(answer(question))

if __name__ == "__main__":
    run()