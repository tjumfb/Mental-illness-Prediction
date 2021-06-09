import efaqa_corpus_zh
import pickle
records = list(efaqa_corpus_zh.load())

data = []
label1 = []
label2 = []
label3 = []
for record in records:
    label1.append(int(record['label']['s1'][-2:].replace('.',''))-1)
    label2.append(int(record['label']['s2'][-1].replace('.',''))-1)
    label3.append(int(record['label']['s3'][-1].replace('.',''))-1)

    content = record['title']
    chats = record['chats']
    for chat in chats:
        if chat['sender'] == 'owner':
            content += ' '+chat['value']
    data.append(content)


# bert-serving-start -model_dir ~/mfb/sentiment_dict/bert/chinese_L-12_H-768_A-12 -num_worker=20 -device_map 0 2 -max_seq_len None
from bert_serving.client import BertClient
from tqdm import tqdm
bc = BertClient()
feature = []
print('Embedding features')
pbar = tqdm(data)
for d in pbar:
    feature.append(bc.encode([d])[0])

l = len(feature)
division_ratio = 4
test_size = l // 4

test_feature = feature[:test_size]
train_feature = feature[test_size+1:]
test_label1 = label1[:test_size]
train_label1 = label1[test_size+1:]
test_label2 = label2[:test_size]
train_label2 = label2[test_size+1:]
test_label3 = label3[:test_size]
train_label3 = label3[test_size+1:]


# print(type(test_feature[0]))
# print(len(test_feature[0]))
# print(type(test_label1[0]))
# print(test_label1[0])
# <class 'numpy.ndarray'>
# 768
# <class 'dict'>
# {'s1': '1.13', 's2': '2.7', 's3': '3.4'}

with open('../data/test_feature.pkl', 'wb') as f:
    pickle.dump(test_feature, f)

with open('../data/train_feature.pkl', 'wb') as f:
    pickle.dump(train_feature, f)

with open('../data/test_label1.pkl', 'wb') as f:
    pickle.dump(test_label1, f)

with open('../data/train_label1.pkl', 'wb') as f:
    pickle.dump(train_label1, f)

with open('../data/test_label2.pkl', 'wb') as f:
    pickle.dump(test_label2, f)

with open('../data/train_label2.pkl', 'wb') as f:
    pickle.dump(train_label2, f)

with open('../data/test_label3.pkl', 'wb') as f:
    pickle.dump(test_label3, f)

with open('../data/train_label3.pkl', 'wb') as f:
    pickle.dump(train_label3, f)

print(label1)
print(label2)
print(label3)