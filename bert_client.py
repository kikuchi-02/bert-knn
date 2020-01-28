from bert_serving.client import BertClient
import numpy as np
import pandas as pd
import os

if __name__== "__main__":
    csv_path = os.path.join('dataset', 'news-en', "abcnews-date-text.csv")
    df = pd.read_csv(csv_path)
    text_list = list(df.iloc[:100, 1])

    bc = BertClient()
    res = bc.encode(text_list)
    res_frame = pd.DataFrame(res)
    res_frame.to_csv('encoded.csv')
