import torch.nn as nn
import torch
from transformers import BertModel
from transformers import BertConfig
from rnn_attention import RNNAttention


class BertPli(nn.Module):
    def __init__(self, model_path, max_para_q, max_para_d, max_len, criterion):
        super(BertPli, self).__init__()
        self.max_para_q = max_para_q
        self.max_para_d = max_para_d
        self.max_len = max_len
        self.criterion = criterion

        # stage 2
        self.config = BertConfig.from_pretrained(model_path, return_dict=False)
        self.bert = BertModel.from_pretrained(model_path, config=self.config)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, self.max_para_q))

        # stage 3
        self.attn = RNNAttention(max_para_d=self.max_para_d)

    def forward(self, data, label, mode='train', pooling='cls'):  # 一个batch数据
        input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']

        # cls : [b * max_para_q * max_para_d, h]
        # last_hidden_state : [b * max_para_q * max_para_d, max len, h]
        last_hidden_state, cls = self.bert(input_ids=input_ids.view(-1, self.max_len),  # [b * max_para_q * max_para_d, max len]
                                            attention_mask=attention_mask.view(-1, self.max_len),  # [b * max_para_q * max_para_d, max len]
                                            token_type_ids=token_type_ids.view(-1, self.max_len))  # [b * max_para_q * max_para_d, max len]

        if pooling == 'cls':
            feature = cls
        else:
            feature = torch.mean(last_hidden_state, dim=1)

        feature = feature.view(self.max_para_q, self.max_para_d, -1)  # [max_para_q, max_para_d, h]

        feature = feature.permute(2, 1, 0)  # [h, max_para_d, max_para_q]

        feature = feature.unsqueeze(0)  # [1, h, max_para_d, max_para_q]
        max_out = self.maxpool(feature)  # [1, h, max_para_d, 1]
        max_out = max_out.squeeze()  # [h, max_para_d]
        max_out = max_out.transpose(0, 1)  # [max_para_d, h]
        max_out = max_out.unsqueeze(0)
        # print(max_out.shape)

        score = self.attn(max_out)  # b,2
        loss = self.criterion(score, label)

        if mode == 'eval' or mode == 'test':
            return score, loss

        return loss