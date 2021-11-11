# BERT-PLI应用到LeCaRD数据集
## 介绍
使用 [bert-pli](https://www.ijcai.org/proceedings/2020/0484.pdf) 模型做法律文档匹配任务，主要为解决文本长度太大的问题。



## 模型

采用了bert-pli模型，该模型的主要做法是对query和doc文本分成n，m个段落，使用bert对和n，m个段落进行拼接交互，得到n*m的交互矩阵，最后通过rnn后做attention得到文本相似度。



### 注意

本项目只是采用了bert-pli的模型，对于训练过程做了修改。bert-pli原论文中，因为有段落和段落相似度的标签，所以bert是单独做fine tune的，即stage2是单独训练的。而LeCaRD数据集只有文档和文档之间的相似度，没有段落的，所以本项目直接对stage2和stage3一起训练。



## 数据集

数据集采用清华开源的 [LeCaRD ](http://www.thuir.cn/group/~mzhang/publications/SIGIR2021-MaYixiao.pdf)

数据已包含在项目中，clone即可使用



LeCaRD/data/candidates 

包含每个问题对应的候选集，对每个问题，候选集大小为100，至少包含一个正样本。



LeCaRD/data/label/golden_labels.json

包含每个问题对应的正确答案



LeCaRD/data/query/query.json 

包含问题的原文以及案由



LeCaRD/data/prediction 

用于存放测试结果



 LeCaRD/metrics.py 

计算测试集指标的代码



LeCaRD/data/prediction/test.json

测试数据



## 预训练模型

pretrained_model/bert-base-chinese 

[bert模型文件](https://huggingface.co/bert-base-chinese/tree/main)，用户自行下载，删去其中tf_model.h5文件



## 代码

### bertpli.py

stage2， bertpli模型



### rnn_attention.py

stage3，通过bert后做的rnn attention操作。



### train.py

训练代码



### test.py

测试代码



### run.sh

运行训练的脚本



### 注意

段落数量不建议修改，大了可能会爆显存。

query平均长度是400+，所以取2段，每段长度小于255

doc最大长度20000+，所以取13段，每段长度小于255，想取更大，会爆显存（24G）。

```python
max_para_q = 2
max_para_d = 13
```

训练一共使用了8张卡跑，每张卡24G显存。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py
```

对应batch size 也是8，即一张卡一次只能跑一条数据。

```python
batch_size = 8
```



## 运行

### 训练

```shell
./run.sh
```

### 测试

```shell
python test.py
```



### 查看结果

```shell
cd LeCaRD
python metrics.py --q test --m NDCG 
python metrics.py --q test --m P
python metrics.py --q test --m MAP
```



## 结果

| 正负样本比例 | lr       | q_para | d_para | para_len | max_len | [P@5](mailto:P@5) | [P@10](mailto:P@10) | MAP    | NDCG10 | NDCG20 | NDCG30 |
| ------------ | -------- | ------ | ------ | -------- | ------- | ----------------- | ------------------- | ------ | ------ | ------ | ------ |
| 1：2         | 2.00E-05 | 2      | 13     | 255      | 512     | 0.55              | 0.47                | 0.6147 | 0.8832 | 0.9016 | 0.9504 |

