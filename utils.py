from bert_serving.client import BertClient
import numpy as np
import pandas as pd
import nmslib
import os
import time
import sentencepiece as spm

def read_data(csv_file_name: str) -> pd.DataFrame:
    dot_index = csv_file_name.find('.')
    file_type = csv_file_name[dot_index+1:]
    if file_type == 'csv':
        df = pd.read_csv(csv_file_name)
    else:
        print('add type:', file_type)
        return None
    return df

def to_piece():
    s = spm.SentencePieceProcessor()
    s.Load('./bert/jp/Japanese_L-12_H-768_A-12_E-30_BPE_WWM/wiki-ja.model')

def bert_encode(text_list: list) -> list:
    print('start encoding')
    start = time.time()
    bc = BertClient()
    vec_list = bc.encode(text_list)
    end = time.time()
    print('finish encoding')
    print(end-start)
    return vec_list

def get_near(vec_list: list, query: list) -> list:
    # initialize a new index, using a HNSW index on Cosine Similarity
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(vec_list)
    # index.addDataPointBatch(data, ids=list(df.iloc[:, 0].values))
    
    index_start = time.time()
    index.createIndex({'post': 2}, print_progress=True)
    index_end = time.time()
    print('finish indexing', index_end-index_start)

    # query for the nearest neighbours of the first datapoint
    query_start = time.time()
    ids, distances = index.knnQuery(query, k=10)
    query_end = time.time()
    print('finish query', query_end-query_start)

    # get all nearest neighbours for all the datapoint
    # using a pool of 4 threads to compute
    # neighbours = index.knnQueryBatch(data, k=10, num_threads=4)

    # print(neighbours)
    print("id:",ids)
    print("distance", distances)
    return ids
