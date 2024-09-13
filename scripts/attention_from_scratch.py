from transformers import BertTokenizer
import torch
import torch.nn as nn
from tqdm import tqdm

# Get the ber tokeninzer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = "Bank of the river."

# Get the tokens by applying the tokenizer to the sentence
tokens = tokenizer.tokenize(sentence)

# Compute the token ids
input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))

# embedding parameters, vocab size of 1000 is a normal baseline number
vocab_size = 10000
embedding_dim = 128

# create the embedding layer
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# apply the embedding layer to the token ids to have a vector representation of the tokens
vector_representations = embedding_layer(input_ids)

print(f'Original token vectors shape: {vector_representations.shape}')


#
#
# REGULAR WAY TO DO IT
#
#

post_attention_vectors = []

for v in tqdm(vector_representations, desc='Computing attention...'):
    # compte the doct product between v and all other vectors
    dot_products = torch.tensor([torch.dot(v, vx) for vx in vector_representations])
    # compute the attention weights for all v.vx pairs
    attention_scores = torch.softmax(dot_products, dim=0)
    # multiply the weights by each of the original vectors and sum them up to get the new vector
    post_attention_vectors.append((attention_scores.view(-1, 1) * vector_representations).sum(dim=0))

post_attention_vectors = torch.stack(post_attention_vectors)

print(f'Post attention token vectors shape: {post_attention_vectors.shape}')

#
#
# UTILIZING K Q V
#
#

K, Q, V = vector_representations, vector_representations, vector_representations

attention_scores = torch.softmax(K @ Q.T, dim=1)
post_attention_vectors_kqv = attention_scores @ V

print(f'Post attention token vectors utilizing KQV shape: {post_attention_vectors_kqv.shape}')