# https://www.kaggle.com/team-ai/shinzo-abe-japanese-prime-minister-twitter-nlp/version/1
from bert_serving.client import BertClient
import numpy as np
import pandas as pd
import os

if __name__== "__main__":
    csv_path = os.path.join('dataset', 'ja', "shinzo_abe.csv")
    abe_df = pd.read_csv(csv_path)
    # abe_new_df = abe_df.loc[:, ['Tweet Nav', 'Tweet Text Size Block']].set_index('Tweet Nav')
    text_list = list(abe_df.loc[:, 'Tweet Text Size Block'])

    bc = BertClient()
    res = bc.encode(text_list)
    res_frame = pd.concat([pd.DataFrame(res), abe_df.loc[:, 'Tweet Nav']], axis=1).set_index('Tweet Nav')
    res_frame.to_csv('abe_encoded.csv')
