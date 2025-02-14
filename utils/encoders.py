#TODO move one hot encoding to this file
#TODO word2vec

from gensim.models import Word2Vec
import numpy as np


def generate_kmers(sequence, k=3):
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]


# Gensim-based Word2Vec model (as an alternative to GloVe)
def train_word2vec(dataset, k=3, vector_size=100, window=5, epochs=10):
    """
    Train a Word2Vec model on DNA sequences (k-mers).

    Args:
        dataset (list of str): List of DNA sequences.
        k (int): Length of k-mers.
        vector_size (int): Embedding size.
        window (int): Context window size.
        epochs (int): Training epochs.

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    # Tokenize into k-mers
    kmers_dataset = [generate_kmers(seq, k) for seq in dataset]

    # Train Word2Vec
    model = Word2Vec(sentences=kmers_dataset, vector_size=vector_size, window=window, min_count=1, sg=1, epochs=epochs)
    return model


# Encode sequences
def encode_sequence(sequence, model, k=3):
    """
    Encode a DNA sequence using a trained Word2Vec model.

    Args:
        sequence (str): DNA sequence to encode.
        model (gensim.models.Word2Vec): Trained Word2Vec model.
        k (int): Length of k-mers.

    Returns:
        np.ndarray: Encoded vector (mean of k-mer embeddings).
    """
    kmers = generate_kmers(sequence, k)
    vectors = [model.wv[kmer] for kmer in kmers if kmer in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)