import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("当前工作目录:", os.getcwd())
print("目录下的文件:", os.listdir('.'))

EN_PATH = "./data/zh-en.en"       # 你的英文文件名, 比如 'train.en'
ZH_PATH = "./data/zh-en.zh"       # 你的中文文件名, 比如 'train.zh'
MAX_LEN = 70              # 句子最大长度
MAX_VOCAB_SIZE = 15000     # 词表大小
BATCH_SIZE = 32

def load_data(en_path,zh_path):
    en_sentences = []
    zh_sentences = []

    with open(en_path,'r',encoding='utf-8')as f_en:
        for line in f_en:
            #英文分割并转成小写
            en_sentences.append(line.strip().lower())

    with open(zh_path,'r',encoding='utf-8')as f_zh:
        for line in f_zh:
            #中文分割
            zh_sentences.append(line.strip())

    #如果不相等，输出后面的句子
    assert len(en_sentences) == len(zh_sentences),"中英文文件行数不相等"

    filtered = []
    for en,zh in zip(en_sentences,zh_sentences):
        #过滤掉超出MAX_LEN的句子
        if 1<len(en.split())<MAX_LEN and 1<len(zh)<MAX_LEN:
            filtered.append((en,zh))
    print(f"原始句子数:{len(en_sentences)},过滤后:{len(filtered)}")

    return filtered 

def build_vocab(sentences,lang_name,max_size=MAX_VOCAB_SIZE):
    tokenized = []
    for sent in sentences:
        if lang_name =='en':
            tokens = sent.split()
        else: 
            tokens = list(sent)
        tokenized.extend(tokens)
    
    counter = Counter(tokenized)

    #取出现次数最多的max_size-4个token
    most_common = counter.most_common(max_size - 4)

    #填入占位，开头，结尾，未知等符
    itos = ['<PAD>','<SOS>','<EOS>','<UNK>'] + [word for word,_ in most_common]
    stoi = {word:idx for idx, word in enumerate(itos)}
    return stoi,itos

def encode_sentence(sentence,stoi,lang,max_len=MAX_LEN):
    if lang =='en':
        tokens = sentence.split()
    else: 
        tokens = list(sentence)

    #把token转换为数字id，未知的置换为<UNK>的id
    ids = [stoi.get(token,stoi['<UNK>'])for token in tokens]
    #在开头与结尾加上<SOS><EOS>的id
    if len(ids) + 2 > max_len:
        ids = ids[:max_len - 2]
    ids = [stoi['<SOS>']] + ids + [stoi['<EOS>']]
    # 填充
    if len(ids) < max_len:
        ids = ids + [stoi['<PAD>']] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

class TranslationDataset(Dataset):
    def __init__(self,data_pairs,en_stoi,zh_stoi):
        self.data = data_pairs
        self.en_stoi = en_stoi
        self.zh_stoi = zh_stoi

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        en_sent,zh_sent = self.data[idx]
        en_tensor = encode_sentence(en_sent,self.en_stoi,'en')
        zh_tensor = encode_sentence(zh_sent,self.zh_stoi,'zh')
        return en_tensor,zh_tensor

def collate_fn(batch):
    en_batch,zh_batch = zip(*batch)
    return torch.stack(en_batch),torch.stack(zh_batch)

if __name__ == "__main__":
    # 加载原始数据
    data_pairs = load_data(EN_PATH, ZH_PATH)
    en_sentences = [pair[0] for pair in data_pairs]
    zh_sentences = [pair[1] for pair in data_pairs]

    # 构建词表
    en_stoi, en_itos = build_vocab(en_sentences, 'en')
    zh_stoi, zh_itos = build_vocab(zh_sentences, 'zh')
    print(f"英文词表大小: {len(en_stoi)}")
    print(f"中文词表大小: {len(zh_stoi)}")

    # 创建Dataset和DataLoader
    dataset = TranslationDataset(data_pairs, en_stoi, zh_stoi)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 测试取一个batch，看看形状
    for en_batch, zh_batch in dataloader:
        print("英文batch形状:", en_batch.shape)   # [batch, max_len]
        print("中文batch形状:", zh_batch.shape)
        break

    print("数据预处理完成！")

    PAD_IDX = en_stoi['<PAD>']
    SOS_IDX = zh_stoi['<SOS>']   
    EOS_IDX = zh_stoi['<EOS>']

    src_tensors = []   # 英文编码张量列表
    tgt_tensors = []   # 中文编码张量列表
    for en_sent, zh_sent in data_pairs:
        src_tensor = encode_sentence(en_sent, en_stoi, 'en')  # 形状 (MAX_LEN,)
        tgt_tensor = encode_sentence(zh_sent, zh_stoi, 'zh')
        src_tensors.append(src_tensor)
        tgt_tensors.append(tgt_tensor)

    src_data = torch.stack(src_tensors)   # (num_samples, MAX_LEN)
    tgt_data = torch.stack(tgt_tensors)   # (num_samples, MAX_LEN)

    save_dict = {
        'src_data': src_data,            # 英文张量
        'tgt_data': tgt_data,            # 中文张量
        'en_stoi': en_stoi,              # 英文词表（str->idx）
        'zh_stoi': zh_stoi,              # 中文词表（str->idx）
        'zh_itos': zh_itos,              # 中文索引列表（idx->str）
        'en_itos': en_itos,
        'MAX_LEN': MAX_LEN,
        'PAD_IDX': PAD_IDX,
        'SOS_IDX': SOS_IDX,
        'EOS_IDX': EOS_IDX,
        'en_vocab_size': len(en_stoi),
        'zh_vocab_size': len(zh_stoi),
    }

    torch.save(save_dict, './data/Trainingdata.pt')
    print("预处理数据已保存到 Trainingdata.pt")

inference_data = {
    'en_stoi': en_stoi,
    'zh_stoi': zh_stoi,
    'zh_itos': zh_itos,
    'MAX_LEN': MAX_LEN,
    'PAD_IDX': PAD_IDX,
    'SOS_IDX': SOS_IDX,
    'EOS_IDX': EOS_IDX,
}
torch.save(inference_data, './data/inference_vocab.pt')
print("字典表已保存到 inference_vocab.pt")

# data = torch.load('Trainingdata.pt')
# tgt_data = data['tgt_data']
# print("第一个样本完整序列（前50个）:", tgt_data[0, :50])
# print("序列中非零元素个数:", (tgt_data[0] != 0).sum().item())
# print("序列中值为 2 (EOS) 的位置:", (tgt_data[0] == 2).nonzero(as_tuple=True)[0])
