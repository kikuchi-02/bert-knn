import nmslib
import numpy as np
import pandas as pd
# create a random matrix to index
# data = np.random.randn(10000, 100).astype(np.float32)

df = pd.read_csv('encoded.csv')
data = df.values.astype(np.fload32)

# initialize a new index, using a HNSW index on Cosine Similarity
index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(data)
index.createIndex({'post': 2}, print_progress=True)

# query for the nearest neighbours of the first datapoint
ids, distances = index.knnQuery(data[0], k=10)

# get all nearest neighbours for all the datapoint
# using a pool of 4 threads to compute
# neighbours = index.knnQueryBatch(data, k=10, num_threads=4)

# print(neighbours)
print(ids)
print(distances)

# https://qiita.com/wasnot/items/20c4f30a529ae3ed5f52
# data
# https://www.kaggle.com/therohk/million-headlines/data#

