from sentence_transformers import SentenceTransformer, LoggingHandler
import logging
import torch
import pickle
import numpy as np

np.set_printoptions(threshold=100)
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO, handlers=[LoggingHandler()])


class VectorGenerator:
    def __init__(self, model_name_or_path="all-distilroberta-v1", use_gpu=True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name_or_path, device=self.device)

    def sentences_to_vectors(self, sentences, batch_size=32, show_progress_bar=True, return_tensors=False):
        return self.model.encode(sentences, batch_size=batch_size, device=self.device,
                                 show_progress_bar=show_progress_bar, convert_to_tensor=return_tensors)

    @staticmethod
    def save_vectors(sentences, vectors, output_path) -> None:
        with open(output_path, 'wb') as f_out:
            pickle.dump({"sentences": sentences, "embeddings": vectors}, f_out, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_vectors(filepath):
        with open(filepath, 'rb') as f_in:
            data = pickle.load(f_in)
            return data["sentences"], data["embeddings"]
