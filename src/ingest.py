import os
import fitz # PyMuPDF
import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions


load_dotenv()

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF and extracts text page by page.
    Returns a list of dictionaries, one per page, with text and metadata.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if text.strip():  # skip blank pages
            pages.append({
                "text": text,
                "metadata": {
                    "source": os.path.basename(pdf_path),
                    "page": page_num + 1
                }
            })
    doc.close()
    return pages

def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Splits page text into overlapping chunks for embedding.
    Preserves source and page metadata for context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])

        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "metadata": {
                    "source": page["metadata"]["source"],
                    "page": page["metadata"]["page"],
                    "chunk_index": i
                }
            })
    return chunks

def get_chroma_collection(collection_name: str = "nuclear_docs"):
    """
    Creates or connects to a persistent ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=".chroma")
    
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    return collection

def ingest_documents(data_folder: str = "data"):
    """
    Main ingestion function. Processes all PDFs in the data folder
    and stores chunks in ChromaDB.
    """
    collection = get_chroma_collection()
    
    pdf_files = [
        f for f in os.listdir(data_folder)
        if f.endswith(".pdf")
    ]
    if not pdf_files:
        print("No PDF files found in data folder.")
        return
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        print(f"Processing {pdf_file}...")

        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_pages(pages)

        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [
            f"{pdf_file}_p{c['metadata']['page']}_c{c['metadata']['chunk_index']}"
            for c in chunks
        ]

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f" Stored {len(chunks)} chunks from {pdf_file}")
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_documents()