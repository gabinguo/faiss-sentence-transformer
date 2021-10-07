from vector_generator.sbert_wrapper import VectorGenerator
from vector_store.faiss_vectorStore import FAISSVectorStore

if __name__ == '__main__':
    path = "/home/guo/faiss-sentence-transformer/messages_emb.pkl"
    vecGen = VectorGenerator(use_gpu=True)
    vecStore = FAISSVectorStore()
    sentences, vectors = vecGen.load_vectors(path)
    vecStore.add_vectors(sentences, vectors)

    query = "I want my pension!!!"
    vecMsg = vecGen.sentences_to_vectors([query], show_progress_bar=False)
    Ds, Is = vecStore.search_vectors(vecMsg, 5)
    for index in Is[0]:
        print(sentences[index], end="\n ***** END ***** \n")