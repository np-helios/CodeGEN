from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import gradio as gr


REPO_PATH = "./"  # Current repository path
COLLECTION_NAME = "code-repository"
OLLAMA_MODEL = "gemma3:12b"  # You can change to any model available in your Ollama instance
BASE_URL = "https://2wv5trgkcfu896-11434.proxy.runpod.net"


embeddings = FastEmbedEmbeddings(model_name="thenlper/gte-large")


client = QdrantClient(path="./qdrant_db")  # Store locally

sample_text = "S"
sample_embedding = embeddings.embed_query(sample_text)


try:
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' already exists")
except (UnexpectedResponse, ValueError):
    # Create new collection with cosine similarity
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=(len(embeddings.embed_query("Nishtha Pandey"))), distance=Distance.COSINE),
    )
    print(f"Created new collection: '{COLLECTION_NAME}'")


vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)


def load_code_repository(repo_path):
    # Define file extensions to include
    code_extensions = [".py", ".js", ".java", ".cpp", ".c", ".h", ".cs", ".go", ".rb", ".php", ".ts", ".html", ".css"]

    # Create a list of glob patterns for each extension
    glob_patterns = [f"**/*{ext}" for ext in code_extensions]

    # Load all code files
    all_files = []
    for pattern in glob_patterns:
        loader = DirectoryLoader(
            repo_path,
            glob=pattern,
            loader_cls=TextLoader,
            show_progress=True
        )
        files = loader.load()
        all_files.extend(files)

    print(f"Loaded {len(all_files)} code files from repository")
    return all_files


def process_documents(documents):
    # Use a code-optimized splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from code repository")
    return chunks


def index_repository():
    documents = load_code_repository(REPO_PATH)
    chunks = process_documents(documents)

    # Add documents to vector store
    vector_store.add_documents(documents=chunks)
    print("Repository indexed successfully")


llm = OllamaLLM(model=OLLAMA_MODEL, base_url=BASE_URL)


prompt_template = """
You are an expert software developer assistant. Use the following pieces of context from the code repository to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


def create_rag_chain():
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
    )

    # Create the RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return rag_chain


def query_code_repository(query, rag_chain=None):
    if rag_chain is None:
        rag_chain = create_rag_chain()

    result = rag_chain({"query": query})

    return {
        "answer": result["result"],
        "source_documents": result.get("source_documents", [])
    }

if __name__ == "__main__":
    index_repository()
    # Reuse the RAG chain from the previous code
    rag_chain = create_rag_chain()

    def process_query(query):
        if not query.strip():
            return "Please enter a valid query."

        result = query_code_repository(query, rag_chain)

        # Format the response
        response = result["answer"]

        # Add source information
        response += "\n\n**Sources:**\n"
        for i, doc in enumerate(result["source_documents"]):
            source_path = doc.metadata.get('source', 'Unknown')
            response += f"\n{i+1}. {source_path}"

        return response

    # Create the Gradio interface
    demo = gr.Interface(
        fn=process_query,
        inputs=gr.Textbox(lines=2, placeholder="Ask a question about your code repository..."),
        outputs="markdown",
        title="Local Code Repository RAG System",
        description="Ask questions about your code repository and get context-aware answers."
    )

    # Launch the interface
    demo.launch(share=False)
