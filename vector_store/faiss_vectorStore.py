import faiss


class FAISSVectorStore:
    def __init__(self, store_type="l2", dim=768):
        self.sentences = []
        if store_type == "l2":
            self.store = faiss.IndexFlatL2(dim)

    def add_vectors(self, sentences, vectors):
        self.sentences += sentences
        self.store.add(vectors)

    def search_vectors(self, query_vector, topk=5):
        D, I = self.store.search(query_vector, topk)
        return D, I
