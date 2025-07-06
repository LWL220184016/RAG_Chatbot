import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(
            self, 
            model_name: str = "neuml/pubmedbert-base-embeddings"
        ):
        
        self.model = SentenceTransformer(model_name, local_files_only=True)

    def embed(self, sentences: list[str]) -> np.ndarray:
        return self.model.encode(sentences)
    
if __name__ == "__main__":
    embedder = Embedder()
    sentences = ["This is an example sentence", "Each sentence is converted to a vector"]
    embeddings = embedder.embed(sentences)
    print(embeddings)
    print(f"Vector size: {embeddings.shape}")