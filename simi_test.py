"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer
import scipy.spatial
import json
import time
import torch
embedder = SentenceTransformer('bert-base-nli-mean-tokens')

kpcorpus = []
#file = open('data/keyphrase/json/kp20k/kp20k_train.json','r')
files_path = ['data/keyphrase/json/kp20k/kp20k_train.json',
              'data/keyphrase/json/kp20k/kp20k_valid.json',
              'data/keyphrase/json/kp20k/kp20k_test.json']
for file_path in files_path:
    file = open(file_path, 'r')
    for line in file.readlines():
        dic = json.loads(line)
        #print(dic)
        kpcorpus.append(dic['title'] + ' ' + dic['abstract'])
        #print(kpcorpus)

# Corpus with example sentences
# corpus = ['A man is eating a food.',
#           'A man is eating a piece of bread.',
#           'The girl is carrying a baby.',
#           'A man is riding a horse.',
#           'A woman is playing violin.',
#           'Two men pushed carts through the woods.',
#           'A man is riding a white horse on an enclosed ground.',
#           'A monkey is playing drums.',
#           'A cheetah is running behind its prey.'
#           ]
num_of_corpusexample = len(kpcorpus)
print(num_of_corpusexample)
time_a = time.time()
corpus_embeddings = embedder.encode(kpcorpus[:num_of_corpusexample])
print("corpus embeddings cost time: ", time.time() - time_a)

# Query sentences:
#queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']
#query_embeddings = embedder.encode(queries)
queries = kpcorpus[:num_of_corpusexample]
query_embeddings = corpus_embeddings
time_a = time.time()
# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 6
similar_docs_matrix = None
i = -1
for query, query_embedding in zip(queries, query_embeddings):
    i += 1
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    #for idx, distance in results[0:closest_n]:
        #print(kpcorpus[idx].strip(), "(Score: %.4f)" % (1-distance))
    idxs = []
    for idx, distance in results[0:closest_n]:
        idxs.append(idx)
    idxs = torch.LongTensor(idxs)
    one_similar = torch.zeros(1, num_of_corpusexample)
    one_similar[0,idxs] = 1
    print("one_hot", one_similar)
    if i == 0:
        similar_docs_matrix = one_similar
    else:
        similar_docs_matrix = torch.cat([similar_docs_matrix, one_similar])
print("semantic search cost time: ", time.time() - time_a)
torch.save(similar_docs_matrix, './data/similar_docs_matrix')