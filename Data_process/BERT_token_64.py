import pandas as pd
import transformers as ppb
import pickle
import numpy as np


# torch.cuda.set_device(0)
# txt_path = r'../data/train.csv'
txt_path = r'../data/val.csv'
# txt_path = r'../data/test.csv'
# save_path = r'../data/BERT_encode/BERT_id_train_64_new.npz'
save_path = r'../data/BERT_encode/BERT_id_val_64_new.npz'
# save_path = r'../data/BERT_encode/BERT_id_test_64_new.npz'
csv_data = pd.read_csv(txt_path, error_bad_lines=False, header=None)
dataset = csv_data[2]

model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
tokenized = dataset.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

max_len = 64
padded = []
for i in tokenized.values:
    if len(i) < max_len:
        i += [0] * (max_len-len(i))
    else:
        i = i[:max_len]
    padded.append(i)
padded = np.array(padded)

print(padded.shape)  # shape;[68108,max_len]
attention_mask = np.where(padded != 0, 1, 0)
print(attention_mask)
print(padded)
print(attention_mask.shape)
print(padded.shape)
print(csv_data[1].shape)
print(csv_data[0].shape)
dict={'caption_id': padded, 'attention_mask': attention_mask, 'images_path': csv_data[1], 'labels': csv_data[0]}
with open(save_path, 'wb') as f:
    pickle.dump(dict, f)
