import torch
import json
import os
from dataset import SimilarLawTestDataSet
from transformers import BertTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

model_path = 'pretrained_model/bert-base-chinese'
candidates_path = 'LeCaRD/data/candidates/candidates'
query_path = 'LeCaRD/data/query/query.json'
golden_label_path = 'LeCaRD/data/label/golden_labels.json'

best_model_path = 'best_model/best_model'

hidden_size = 768
max_para_q = 2
max_para_d = 13
para_max_len = 255
max_len = 512
batch_size = 1
gradient_accumulation_steps = 10

model = torch.load(best_model_path)

tokenizer = BertTokenizer.from_pretrained(model_path)
criterion = nn.CrossEntropyLoss()

dataset = SimilarLawTestDataSet(candidates_path, query_path, golden_label_path, tokenizer,
                                max_para_q, max_para_d, para_max_len, max_len)
test_data_loader = DataLoader(dataset, batch_size=batch_size)


def test():
    model.eval()
    with torch.no_grad():
        batch_iterator = tqdm(test_data_loader, desc='testing...')
        total_step = len(batch_iterator)
        total_loss = 0.0
        num = 0
        corr = 0
        ans_dict = {}
        for step, (query_id, doc_id, input_ids, token_type_ids, attention_mask) in enumerate(batch_iterator):
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()
            label = torch.ones([1], dtype=torch.int64).cuda()

            data = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            score, loss = model(data, label, 'test')
            is_pos = score[:, 1]
            max_score, max_idx = torch.max(score, dim=1)
            num += max_idx.size()[0]

            loss = loss.mean()
            total_loss += loss.item()

            for qid, did, s in zip(query_id, doc_id, is_pos):
                if qid not in ans_dict.keys():
                    ans_dict[qid] = {}
                ans_dict[qid][did] = s.item()

        for k in ans_dict.keys():
            ans_dict[k] = sorted(ans_dict[k].items(), key=lambda x: x[1], reverse=True)

        for k in ans_dict.keys():
            ans_dict[k] = [int(did) for did, _ in ans_dict[k]]

        with open('LeCaRD/data/prediction/res.json', mode='w', encoding='utf-8') as f:
            f.write(json.dumps(ans_dict, ensure_ascii=False))
    print("test finish")
    return total_loss / total_step, corr / num

if __name__ == '__main__':
    test()