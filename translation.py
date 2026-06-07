import torch
import torch.nn as nn
import math
from model import TransformerModel
from model import TransformerModel, PositionalEncoding

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("当前工作目录:", os.getcwd())
print("目录下的文件:", os.listdir('.'))

vocab_data = torch.load('./data/inference_vocab.pt', map_location='cpu')
en_stoi = vocab_data['en_stoi']
zh_stoi = vocab_data['zh_stoi']
zh_itos = vocab_data['zh_itos']
MAX_LEN = vocab_data['MAX_LEN']
PAD_IDX = vocab_data['PAD_IDX']
SOS_IDX = vocab_data['SOS_IDX']
EOS_IDX = vocab_data['EOS_IDX']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(len(en_stoi), len(zh_stoi),PAD_IDX).to(device)
model.load_state_dict(torch.load('./PostTrain/best_model.pth', map_location=device))
model.eval()

def encode_sentence(sentence, stoi, max_len=MAX_LEN):
    tokens = sentence.lower().split()

    ids = [stoi.get(token,stoi['<UNK>']) for token in tokens]

    # if len(ids) > max_len:
    #     ids = ids[:max_len]
    # else:
    #     ids = ids + [stoi['<PAD>']] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0) 

def greedy_decode(model, src_tensor, max_len=MAX_LEN):
    src_tensor = src_tensor.to(device)
    tgt_ids = [SOS_IDX]
    for step in range(max_len):
        tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
        # 直接调用模型，模型内部会编码 src 并解码
        output = model(src_tensor, tgt_tensor)   # (1, tgt_len, vocab)
        next_token = output[0, -1, :].argmax().item()
        print(
            step,
            next_token,
            zh_itos[next_token]
        )
        if next_token == EOS_IDX:
            break
        tgt_ids.append(next_token)
    return tgt_ids[1:]
    
def translate(model, src_sentence, max_len=50):
    model.eval()
    src = torch.tensor(src_sentence).unsqueeze(0).to(device)
    tgt = torch.tensor([SOS_IDX]).unsqueeze(0).to(device)
    for _ in range(max_len):
        out = model(src, tgt)
        next_token = out[:, -1, :].argmax(-1).unsqueeze(1)
        tgt = torch.cat([tgt, next_token], dim=1)
        if next_token.item() == EOS_IDX:
            break
    return tgt.squeeze().tolist()

if __name__ == '__main__':
    while True:
        sent = input("输入英文句子: ")
        if sent == 'quit':
            break
        src_ids = [en_stoi.get(token, en_stoi['<UNK>']) for token in sent.lower().split()]
        translated_ids = translate(model, src_ids)
        translated_sentence = "".join([zh_itos[i] for i in translated_ids if i not in [PAD_IDX, SOS_IDX, EOS_IDX]])
        print("翻译结果:", translated_sentence)