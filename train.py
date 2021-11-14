from transformers import BertTokenizer
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import SimilarLawDataSet
from tqdm import tqdm
from torch.optim import Adam
import os
from bertpli import BertPli
import random
import numpy as np

def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

seed_torch()

model_path = 'pretrained_model/bert-base-chinese'
candidates_path = 'LeCaRD/data/candidates'
query_path = 'LeCaRD/data/query/query.json'
golden_label_path = 'LeCaRD/data/label/golden_labels.json'
best_model_name = 'best_model'
pooling = 'mean'

hidden_size = 768
# 以下两个参数不建议修改，大了可能会爆显存
max_para_q = 2
max_para_d = 13
para_max_len = 255
max_len = 512

epoch = 20
batch_size = 4
learning_rate = 2e-5

gradient_accumulation_steps = 10

tokenizer = BertTokenizer.from_pretrained(model_path)
criterion = nn.CrossEntropyLoss()

model = BertPli(model_path=model_path, max_para_q=max_para_q, max_para_d=max_para_d, max_len=max_len, criterion=criterion)

model.cuda()

if torch.cuda.device_count() > 1:
    print(f"GPU数：{torch.cuda.device_count()}")
    model = nn.DataParallel(model)

optimizer = Adam(model.parameters(), lr=learning_rate)
dataset = SimilarLawDataSet(candidates_path, query_path, golden_label_path, tokenizer, max_para_q, max_para_d, para_max_len, max_len)

train_size = int(0.9*len(dataset))
eval_size = len(dataset) - train_size

train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)


def evaluate():
    model.eval()
    with torch.no_grad():
        batch_iterator = tqdm(eval_data_loader, desc='evaluating')
        total_step = len(batch_iterator)
        total_loss = 0.0
        num = 0
        corr = 0
        for step, (input_ids, token_type_ids, attention_mask, label) in enumerate(batch_iterator):
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()
            label = label.cuda()

            data = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            score, loss = model(data=data, label=label, mode='eval', pooling=pooling)

            _, max_idx = torch.max(score, dim=1)

            batch_corr = (label == max_idx).sum()  # tensor
            corr += batch_corr.item()
            num += max_idx.size()[0]

            loss = loss.mean()
            total_loss += loss.item()

        print("evaluate loss: %f" % (total_loss / total_step))
        print("acc: %f" % (corr / num))
    print("evaluate finish")
    return total_loss/total_step, corr/num


def train():
    best_acc = 0.0
    for e in range(epoch):
        print(f"epoch:{e}")
        step_loss = 0.0
        epoch_loss = 0.0
        batch_iterator = tqdm(train_data_loader, desc='training')
        total_step = len(batch_iterator)
        model.train()
        model.zero_grad()
        for step, (input_ids, token_type_ids, attention_mask, label) in enumerate(batch_iterator):

            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()
            label = label.cuda()

            data = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            loss = model(data=data, label=label, mode='train', pooling=pooling)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            model.zero_grad()

            step_loss += loss.item()
            epoch_loss += loss.item()

            if (step+1) % gradient_accumulation_steps == 0:
                print("avg loss:", step_loss/gradient_accumulation_steps)
                step_loss = 0.0

        epoch_loss /= total_step
        print("epoch %d loss: %f" % (e, epoch_loss))
        eval_loss, acc = evaluate()
        if acc > best_acc:
            best_acc = acc
            torch.save(model, 'best_model/best_model')
    print("train finish")


if __name__ == '__main__':
    train()