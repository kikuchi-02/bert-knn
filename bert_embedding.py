import torch
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
from transformers import BertModel
import time
import os

class BertEncoder:
    def __init__(self):
        path = './bert_pretrained'
        cache = 'bert-base-japanese-whole-word-masking'
        if os.path.isdir(path):
            model_dir = path
            self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_dir)
            self.model = BertModel.from_pretrained(model_dir)
        else:
            os.mkdir(path)
            model_dir = cache
            self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_dir)
            self.model = BertModel.from_pretrained(model_dir)
            self.save_model()

    def save_model(self):
        self.model.save_pretrained('bert_pretrained')
        self.tokenizer.save_vocabulary('bert_pretrained')

    def encode_text(self, text: str):
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(input_ids)
            last_hidden = outputs[0]
        mean_last_hidden = torch.mean(last_hidden[0], dim=0)
        return mean_last_hidden

    def test_time(self):
        start = time.time()
        text = '青葉山 で 植物 の 研究 を し て い ます 。'
        self.encode_text(text)
        end = time.time()
        print(end-start)


if __name__ == '__main__':
    encoder = BertEncoder()
    encoder.test_time()
    # encoder.save_model()
