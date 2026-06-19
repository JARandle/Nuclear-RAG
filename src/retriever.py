import chromadb
from chromadb.utils import embedding_functions

def get_chroma_collection(collection_name: str = "nuclear_docs"):
    """
    Connects to the existing persistent ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=".chroma")
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    return collection

def retrieve_chunks(query: str, n_results: int = 5) -> list[dict]:
    """
    Takes a natural language query and returns the top matching chunks from ChromaDB along with their metadata.
    """
    collection = get_chroma_collection()

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    return chunks

if __name__ == "__main__":
    test_query = "What are the requirements for HEPA filter testing?"
    results = retrieve_chunks(test_query)

    print(f"Query: {test_query}\n")
    for i, chunk in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Source: {chunk['metadata']['source']}, Page {chunk['metadata']['page']}")
        print(f"  Distance: {chunk['distance']:.4f}")
        print(f"  Text preview: {chunk['text'][:100]}...")
        print()