import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import itertools

def create_embedding_matrix(wv, vector_dim):
    vocab_size = len(wv.vocab)
    embedding_matrix = np.zeros((vocab_size+2, vector_dim))
    for i in range(vocab_size+2):
        if i == 1:
            embedding_matrix[i] = np.ones(vector_dim)
        elif i > 1:
            embedding_vector = wv[wv.index2word[i-2]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

# we reserve 0 - padding and 1 a token that is not in the vocabulary
# if the embedding is frozen it would be enough to have only zero preserved for padding 
def tokens2index(tokens, wv):
    result = []
    for token in tokens:
        if token in wv:
            result.append(wv.vocab[token].index+2)
        else:
            result.append(1)
            print(f"'{token}' is not in the vocabulary - '{' '.join(tokens)}'")
    return result

def texts2index_padded(texts, wv, seq_length):
    texts_ids = [tokens2index(tokens, wv) for tokens in texts]
    return pad_sequences(texts_ids, maxlen=seq_length, 
                          dtype='int32', padding='post', truncating='post', value=0.0)

def prune_wv(wv, tokenized_names):
    words_to_keep = list(set(list(itertools.chain.from_iterable(tokenized_names))))
    words_to_trim = [w for w in wv.vocab.keys() if w not in words_to_keep]
    ids_to_trim = [wv.vocab[w].index for w in words_to_trim]

    for w in words_to_trim:
        del wv.vocab[w]

    wv.vectors = np.delete(wv.vectors, ids_to_trim, axis=0)
    wv.init_sims(replace=True)

    for i in sorted(ids_to_trim, reverse=True):
        del(wv.index2word[i])

    for i in range(len(wv.vocab.keys())):
        word = wv.index2word[i]
        wv.vocab[word].index = i
        
    return wv
    