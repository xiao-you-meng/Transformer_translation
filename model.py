import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader,TensorDataset
import math
import time
from colorama import init, Fore, Back, Style
import sys
import csv

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5000, batch_first=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        else:
            pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,pad_idx,d_model=128,nhead=4,num_encoder_layers=3,num_decoder_layers=3,dim_feedforward=512,dropout=0):
        super(TransformerModel,self).__init__()
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers,
            dim_feedforward, dropout, batch_first=True   # 改为 True
            )
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size,d_model,padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size,d_model,padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model,dropout,batch_first=True)
        #self.transformer = nn.Transformer(d_model,nhead,num_encoder_layers,num_decoder_layers,dim_feedforward,dropout)
        self.generator = nn.Linear(d_model,tgt_vocab_size)
        #self._init_parameters()
        self.pad_idx = pad_idx

    def forward(self,src,tgt,src_mask=None,tgt_mask=None,src_padding_mask=None,tgt_padding_mask=None,memory_key_padding_mask=None):
        src_emb = self.src_embedding(src)*math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt)*math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        src_padding_mask = (src == self.pad_idx)
        tgt_padding_mask = (tgt == self.pad_idx)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1)
        ).to(src.device)

        #memory = self.transformer.decoder(src_emb,mask=src_mask,src_key_padding_mask=src_padding_mask)
        #output = self.transformer.decoder(tgt_emb,memory,tgt_mask=tgt_mask,memory_mask=None,tgt_key_padding_mask=tgt_padding_mask,memory_key_padding_mask=memory_key_padding_mask)
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        return self.generator(output)

if __name__ == "__main__":

    f_csv = open('./Seq2Seq/log/training_log.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(f_csv)
    writer.writerow(['epoch','loss','accuracy','prediction'])

    log_file = open("./Seq2Seq/log/training_log.txt", "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file  # 如果想连错误信息也保存

    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("当前工作目录:", os.getcwd())
    print("目录下的文件:", os.listdir('.'))

    data = torch.load('./data/Trainingdata.pt')
    src_data = data['src_data']
    tgt_data = data['tgt_data']
    en_stoi = data['en_stoi']
    zh_stoi = data['zh_stoi']
    zh_itos = data['zh_itos']
    en_itos = data['en_itos']
    MAX_LEN = data['MAX_LEN']
    PAD_IDX = data['PAD_IDX']
    SOS_IDX = data['SOS_IDX']
    EOS_IDX = data['EOS_IDX']
    en_vocab_size = data['en_vocab_size']
    zh_vocab_size = data['zh_vocab_size']

    # print("zh_itos[:10]:", zh_itos[:10])
    # print("PAD_IDX:", data['PAD_IDX'])
    # print("SOS_IDX:", data['SOS_IDX'])
    # print("EOS_IDX:", data['EOS_IDX'])

    print('GPU avaliable:',torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

################################
#数据集在这里！！！！！！！！！！
#
#
#
    dataset = TensorDataset(src_data,tgt_data)
    #dataset = TensorDataset(src_data[:256], tgt_data[:256])

    dataloader = DataLoader(dataset,batch_size=32,shuffle=True)
    print(len(dataset))

    print(en_vocab_size)

    print(zh_vocab_size)

    print(MAX_LEN)

    model = TransformerModel(en_vocab_size,zh_vocab_size,PAD_IDX)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device:{device}")

    #criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    # optimizer = optim.Adam(model.parameters(),lr=0.01)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    optimizer = AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)

    # Noam warmup scheduler
    # def noam_schedule(step, warmup_steps=4000, d_model=256):
    #     step = step + 1
    #     return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    # scheduler = LambdaLR(optimizer, lr_lambda=noam_schedule)

    #from torch.optim.lr_scheduler import LambdaLR

    # def warmup_lambda(step, warmup_steps=4000, d_model=256):
    #     step = step + 1
    #     return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    # scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    print("PAD_IDX =", PAD_IDX)
    print("SOS_IDX =", SOS_IDX)
    print("EOS_IDX =", EOS_IDX)

    train_start_time = time.time()

    best_loss = float("inf")

    EPOCHS = 75
    print('训练开始！')
    for epoch in range(EPOCHS):
        prediction_str = ''
        print(Fore.RED+"#=========================================#")
        init(autoreset=True)
        print(f'Epoch{epoch+1}:')
        model.train()
        train_loss = 0.0
        epoch_start_time = time.time()
        total_correct = 0
        total_tokens = 0

        for step, (src_batch, tgt_batch) in enumerate(dataloader):
        #for src_batch,tgt_batch in dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            decoder_input = tgt_batch[:,:-1]
            target = tgt_batch[:,1:]

            ####################
            #输出原始句子和对应的中文翻译
            #
            #

            if step == 0:
                src_indices = src_batch[0].cpu().tolist()
                tgt_indices = tgt_batch[0].cpu().tolist()
                src_words = [en_itos[i] for i in src_indices if i not in [PAD_IDX, SOS_IDX, EOS_IDX]]
                tgt_words = [zh_itos[i] for i in tgt_indices if i not in [PAD_IDX, SOS_IDX, EOS_IDX]]
                print("Source sentence (English):", " ".join(src_words))
                print("Target sentence (Chinese):", "".join(tgt_words))
            
            optimizer.zero_grad()
            output = model(src_batch,decoder_input)

            with torch.no_grad():
                pred = output.argmax(-1)

                non_pad_mask = (target != PAD_IDX)

                correct = (pred == target) & non_pad_mask

                total_correct +=correct.sum().item()
                total_tokens += non_pad_mask.sum().item()

                den = non_pad_mask.sum().float()
                acc = (
                    correct.sum().float()
                    / non_pad_mask.sum().float()
                )

                if step == 0:
                    print(
                        f"Accuracy:{acc.item():.4f}"
                    )

            if step == 0:
                print(
                    "non-pad ratio:",
                    (target != PAD_IDX).float().mean().item()
                )

            ####################
            #输出预测句子和原本句子，用来判断训练效果1
            #
            #

            if step == 0:
                with torch.no_grad():

                    pred_tokens = output.argmax(-1)[0]

                    pred_sentence = []

                    for token in pred_tokens:

                        token = token.item()

                        if token == EOS_IDX:
                            break

                        if token not in [PAD_IDX, SOS_IDX]:
                            pred_sentence.append(
                                zh_itos[token]
                            )

                    print(
                        "Prediction:",
                        "".join(pred_sentence)
                    )
            
            if step == 0:
                pred_tokens = output.argmax(-1)[0].detach().cpu().tolist()

                prediction_str = "".join([
                    zh_itos[t]
                    for t in pred_tokens
                    if t not in [PAD_IDX, SOS_IDX, EOS_IDX]
                    and t != EOS_IDX
                ])            

            output = output.reshape(-1,zh_vocab_size)
            target = target.reshape(-1)
            loss = criterion(output,target)
            loss.backward()

            if step == 0:

                total_norm = 0

                for p in model.parameters():

                    if p.grad is not None:

                        total_norm += (
                            p.grad.norm(2).item()
                        ) ** 2

                total_norm = total_norm ** 0.5

                print(
                    f"Gradient Norm:{total_norm:.4f}"
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()
            #scheduler.step()

            # total_norm = 0
            # for p in model.parameters():
            #     if p.grad is not None:
            #         total_norm += p.grad.norm(2).item() ** 2
            # total_norm = total_norm ** 0.5
            # print(f"Gradient norm: {total_norm:.4f}")

            # torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            # optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss/len(dataloader)
        avg_accuracy = total_correct / total_tokens
        writer.writerow([epoch+1, avg_loss, avg_accuracy, prediction_str])

        print(f"Epoch{epoch+1}/{EPOCHS}|Loss:{avg_loss:.4f}")
        print(
            "Current LR:",
            optimizer.param_groups[0]["lr"]
        )
        epoch_end_time = time.time()        # 记录 epoch 结束时间
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch+1} 耗时: {epoch_duration:.2f} 秒")
        torch.save(
            model.state_dict(),
            "./PostTrain/latest.pth"
        )
        if avg_loss < best_loss:
            best_loss = avg_loss

            torch.save(
                model.state_dict(),
                "./PostTrain/best_model.pth"
            )

    log_file.close()
    f_csv.close()
    train_end_time = time.time()
    train_duration = (train_end_time - train_start_time) / 60
    print("训练完成！")
    print(f'训练总耗时:{train_duration:.2f}分钟')
    

    torch.save(model.state_dict(), './PostTrain/transformer_translation.pth')
    print("模型已保存为 transformer_translation.pth")
