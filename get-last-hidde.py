import torch
from transformers import BertModel
import MeCab
from transformers import BertJapaneseTokenizer
tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-whole-word-masking')
text = "テスト"
tokenized_text = [tokenizer.encode(text)]
input_ids = torch.tensor(tokenized_text)
segment_ids = [1] * len(tokenized_text)
segment_tensor = torch.tensor([segment_ids])
print(input_ids)
model = BertModel.from_pretrained('bert-base-japanese-whole-word-masking')
print('loaded')
#with torch.no_grad():
#    output, _ = model(input_ids, segment_tensor)
#    last_hiddden_states = output
model.eval()
last_hiddden_states, _ = model(input_ids, segment_tensor)
print(_.shape)
print(last_hiddden_states)
print(last_hiddden_states.shape)
sentence_embedding = torch.mean(last_hiddden_states, dim=0)
print(sentence_embedding)
