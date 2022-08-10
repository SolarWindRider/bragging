from nrclex import NRCLex
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Linear, Dropout, ReLU, Parameter, LayerNorm, Sequential, MSELoss, KLDivLoss, \
    CrossEntropyLoss, BCELoss, Sigmoid, Tanh, GELU
from torch.nn import functional as F
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
import random
import numpy as np
from collections import Iterator
from math import ceil

# ----------------------------没有验证集----------------------------------------
import conf


class MyDataset(Dataset):
    def __init__(self, tokenizer, conf, istrain=True):
        """
        :param tokenizer: 传入已经实例化完成的tokenizer
        :param conf: obj 传入配置文件
        :param istrain: bool 是否载入训练数据
        """
        self.tokenizer = tokenizer
        self.conf = conf
        # 读取数据
        df = pd.read_csv(self.conf.DATAPATH)
        if istrain:  # 根据论文，trainset是keyword查找到的推特
            data__ = df[df["sampling"] == "keyword"]
            self.data = data__[["text", "label"]].to_numpy()
        else:  # 根据论文，testset是随机查找到的推特
            data__ = df[df["sampling"] == "random"]
            self.data = data__[["text", "label"]].to_numpy()
        if self.conf.CLASSNUM == 2:
            self.label_map = conf.BIN_CLASS_MAP
        elif self.conf.CLASSNUM == 7:
            self.label_map = conf.MULTI_CLASS_MAP
        else:
            raise ValueError("numclass shoud be 2 or 7")

        if self.conf.LingFeature == "Cluster":
            pass  # TODO

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        label = self.label_map[self.data[item][1]]
        inputs = self.tokenizer(self.data[item][0], return_tensors="pt",
                                max_length=self.conf.MAXLENGTH, padding="max_length",
                                truncation=True)
        if self.conf.LingFeature is not None:
            inputs["input_feature"] = get_ling_feature(self.data[item][0], self.conf.LingFeature)
        return inputs, label


def get_ling_feature(text, lingfeature) -> torch.Tensor:
    outputs = None
    if lingfeature == "NRC":
        f_d = NRCLex(text).affect_frequencies
        if "anticipation" in f_d.keys():
            f_d['anticip'] = f_d.pop("anticipation")
        _ = [v for v in f_d.values()]
        outputs = torch.tensor(_)
    elif lingfeature == "Cluster":
        pass  # TODO
    elif lingfeature == "LIWC":
        pass  # TODO
    else:
        raise ValueError("wrong feature type")
    return outputs


# --------------------------bragging原文的模型复现 - -----------------------------------------------------


class SimpleBertModel(Module):
    def __init__(self, vocab_size, conf):
        super().__init__()
        self.conf = conf
        if self.conf.LingFeature is not None:
            self.embedding_lm = AutoModelForMaskedLM.from_pretrained(self.conf.LM,
                                                                     output_hidden_states=True)
            self.attn_gate = AttnGating(self.conf)
        self.lm = AutoModelForMaskedLM.from_pretrained(self.conf.LM)
        if self.conf.LM == "vinai/bertweet-base":
            vocab_size += 1
        self.linear = Linear(vocab_size, self.conf.CLASSNUM)
        self.dropout = Dropout(self.conf.DROPOUT)

    def forward(self, inputs):
        if self.conf.LingFeature is not None:
            _ = self.embedding_lm(input_ids=inputs["input_ids"],
                                  token_type_ids=inputs["token_type_ids"],
                                  attention_mask=inputs["attention_mask"])
            roberta_embed = _[1][0]
            combine_embed = self.attn_gate(roberta_embed, inputs["input_feature"])
            lm_output = self.lm(input_ids=None, token_type_ids=inputs["token_type_ids"],
                                attention_mask=inputs["attention_mask"], inputs_embeds=combine_embed)
        else:
            lm_output = self.lm(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"],
                                attention_mask=inputs["attention_mask"])
        cls = lm_output.logits[:, 0, :]
        if self.conf.LMoutput:
            return cls

        outputs = self.dropout(cls)
        outputs = torch.tanh(outputs)
        outputs = self.dropout(outputs)
        linear_output = self.linear(outputs)
        return linear_output


class AttnGating(Module):
    def __init__(self, conf):
        super(AttnGating, self).__init__()
        self.conf = conf
        self.hidden_size = 768  # 与bert一致
        self.embedding_size = 200  # 与原文一致
        self.dropout = 0.5  # 与原文一致
        linear_input_dim = None
        if self.conf.LingFeature == "LIWC":
            linear_input_dim = 93
        elif self.conf.LingFeature == "NRC":
            linear_input_dim = 10
        elif self.conf.LingFeature == "Cluster":
            linear_input_dim = 200
        self.linear = Linear(linear_input_dim, self.embedding_size)
        self.relu = ReLU(inplace=True)

        self.weight_emotion_W1 = Parameter(torch.Tensor(self.hidden_size + self.embedding_size, self.hidden_size))
        self.weight_emotion_W2 = Parameter(torch.Tensor(self.embedding_size, self.hidden_size))

        torch.nn.init.uniform_(self.weight_emotion_W1, -0.1, 0.1)
        torch.nn.init.uniform_(self.weight_emotion_W2, -0.1, 0.1)

        self.LayerNorm = LayerNorm(self.hidden_size)
        self.dropout = Dropout(self.dropout)

    def forward(self, embeddings_roberta, linguistic_feature):
        # Project linguistic representations into vectors with comparable size
        linguistic_feature = self.linear(linguistic_feature)
        emotion_feature = linguistic_feature.repeat(self.conf.MAXLENGTH, 1, 1)  # (50, bs, 200)
        emotion_feature = emotion_feature.permute(1, 0, 2)  # (bs, 50, 200)

        # Concatnate word and linguistic representations
        features_combine = torch.cat((emotion_feature, embeddings_roberta), axis=2)  # (bs, 50, 968)

        g_feature = self.relu(torch.matmul(features_combine, self.weight_emotion_W1))

        # Attention gating
        H = torch.mul(g_feature, torch.matmul(emotion_feature, self.weight_emotion_W2))
        alfa = min(self.conf.Beta * (torch.norm(embeddings_roberta) / torch.norm(H)), 1)
        E = torch.add(torch.mul(alfa, H), embeddings_roberta)

        # Layer normalization and dropout
        embedding_output = self.dropout(self.LayerNorm(E))

        return embedding_output


# ----------------------绘制loss图像，观察损失下降-----------------------
def get_2loss_fig(loss_train_path="loss_train.pt", loss_dev_path="loss_dev.pt"):
    loss_train_li = pickle.load(open(loss_train_path, "rb"))
    loss_dev_li = pickle.load(open(loss_dev_path, "rb"))
    dic = {
        "epoch": range(len(loss_train_li)),
        "loss_train": loss_train_li,
        "loss_dev": loss_dev_li,
    }
    df = pd.DataFrame(dic)
    sns.set_theme(style="darkgrid")
    sns.lineplot(x='epoch', y='loss_train', data=df, lw=3)
    sns.lineplot(x='epoch', y='loss_dev', data=df, lw=3)
    plt.legend(labels=['loss_train', 'loss_dev'], facecolor='white')
    plt.show()


def get_loss_fig(loss_train_path="loss_train.pt"):
    loss_train_li = pickle.load(open(loss_train_path, "rb"))
    dic = {
        "epoch": range(len(loss_train_li)),
        "loss_train": loss_train_li,
    }
    df = pd.DataFrame(dic)
    sns.set_theme(style="darkgrid")
    sns.lineplot(x='epoch', y='loss_train', data=df, lw=3)
    plt.legend(labels=['loss_train'], facecolor='white')
    plt.show()


# ------------设置seed-----------------------------------------------------------------
def seed_all(seed_value):
    # random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu vars

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------------------#
#                         feature disentangle model utils                               #
# --------------------------------------------------------------------------------------#

# ----------------------------------每次采样，每种类别各采一个batch-------------------------
class GanData(Dataset):
    def __init__(self, tokenizer, conf, istrain, label=None):
        """
        :param tokenizer: 传入已经实例化完成的tokenizer
        :param conf: obj 传入配置文件
        :param istrain: bool 是否载入训练数据
        :param label: str 训练数据的具体类别， 只有istrain为True时才需要传入这个参数
        """
        self.tokenizer = tokenizer
        self.conf = conf
        self.istrain = istrain
        # 读取数据
        df = pd.read_csv(self.conf.DATAPATH)
        if istrain is True:
            self.data = df[(df["sampling"] == "keyword") & (df["label"] == label)][["text", "label"]].values
        elif istrain is False:
            self.data = df[df["sampling"] == "random"][["text", "label"]].values

    def __len__(self):
        length = None
        if self.istrain is True:  # 训练集label为not类别的样本数量,并针对batchsize进行圆整
            length = ceil(2838 * self.conf.EPOCHS / self.conf.BATCHSIZE) * self.conf.BATCHSIZE
        elif self.istrain is False:
            length = len(self.data)
        return length

    def __getitem__(self, item):
        if self.istrain is True:
            inputs = self.tokenizer(self.data[item % len(self.data)][0], return_tensors="pt",
                                    max_length=self.conf.MAXLENGTH, padding="max_length",
                                    truncation=True)
            inputs = inputs.to(self.conf.DEVICE)
            return inputs
        elif self.istrain is False:
            label = torch.tensor(self.conf.MULTI_CLASS_MAP[self.data[item % len(self.data)][1]])
            inputs = self.tokenizer(self.data[item % len(self.data)][0], return_tensors="pt",
                                    max_length=self.conf.MAXLENGTH, padding="max_length",
                                    truncation=True)
            inputs = inputs.to(self.conf.DEVICE)
            return inputs, label  # 注意，label不放在GPU里面


class GanTrainDataLoader(Iterator):
    def __init__(self, tokenizer, conf):
        self.dataloaders = []
        for k in conf.MULTI_CLASS_MAP.keys():
            self.dataloaders.append(
                iter(DataLoader(GanData(tokenizer, conf, istrain=True, label=k), batch_size=conf.BATCHSIZE,
                                shuffle=True)))

    def __next__(self):
        output = []
        for loader in self.dataloaders:
            output.append(next(loader))
        return output

    def __len__(self):
        return ceil(2838 * conf.EPOCHS / conf.BATCHSIZE)


# -----------------------每次采样，一个batch为32，28个----------------------------------------------
class GanDataSP2(Dataset):
    def __init__(self, tokenizer, conf, istrain):
        """
        :param tokenizer: 传入已经实例化完成的tokenizer
        :param conf: obj 传入配置文件
        :param istrain: bool 是否载入训练数据
        :param label: str 训练数据的具体类别， 只有istrain为True时才需要传入这个参数
        """
        self.tokenizer = tokenizer
        self.conf = conf
        self.istrain = istrain
        # 读取数据
        df = pd.read_csv(self.conf.DATAPATH)
        self.df = df.sample(frac=1.0).reset_index(drop=True)

        if istrain is True:
            self.data = []
            data0 = self.df[(self.df["sampling"] == "keyword") & (self.df["label"] == "not")][
                ["text", "label"]].values
            data_ = self.df[(self.df["sampling"] == "keyword") & (self.df["label"] != "not")][
                ["text", "label"]].values
            idx_ = 0
            for idx in range(len(data0)):
                if idx % 28 == 0:
                    for i in range(4):
                        self.data.append(data_[idx_ % len(data_)])
                        idx_ += 1
                    self.data.append(data0[idx])
                else:
                    self.data.append(data0[idx])
        elif istrain is False:
            self.data = df[df["sampling"] == "random"][["text", "label"]].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.istrain is True:
            label = torch.tensor(self.conf.MULTI_CLASS_MAP[self.data[item % len(self.data)][1]])
            inputs = self.tokenizer(self.data[item % len(self.data)][0], return_tensors="pt",
                                    max_length=self.conf.MAXLENGTH, padding="max_length",
                                    truncation=True)
            return inputs, label
        elif self.istrain is False:
            label = torch.tensor(self.conf.MULTI_CLASS_MAP[self.data[item % len(self.data)][1]])
            inputs = self.tokenizer(self.data[item % len(self.data)][0], return_tensors="pt",
                                    max_length=self.conf.MAXLENGTH, padding="max_length",
                                    truncation=True)
            return inputs, label  # 注意，label不放在GPU里面


class BertClsLayer(Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.lm = AutoModelForMaskedLM.from_pretrained(self.conf.LM)

    def forward(self, inputs):
        lm_output = self.lm(input_ids=inputs["input_ids"].squeeze(1),
                            token_type_ids=inputs["token_type_ids"].squeeze(1),
                            attention_mask=inputs["attention_mask"].squeeze(1))
        cls = lm_output.logits[:, 0, :]  # shape (batch, vocabsize)

        return cls


class Generator(Module):
    def __init__(self, vocab_size, conf):
        super().__init__()
        self.conf = conf
        self.bert_cls_layer = BertClsLayer(self.conf)
        self.linear = Linear(vocab_size, self.conf.FeatureDim)
        self.dropout = Dropout(self.conf.DROPOUT)

        self.mlp_c = Sequential(Linear(self.conf.FeatureDim, 1024), Tanh(), Linear(1024, 512), Tanh(), Dropout(),
                                Linear(512, 1024), Tanh(), Dropout(), Linear(1024, self.conf.FeatureDim))
        self.mlp_t = Sequential(Linear(self.conf.FeatureDim, 1024), Tanh(), Linear(1024, 512), Tanh(), Dropout(),
                                Linear(512, 1024), Tanh(), Dropout(), Linear(1024, self.conf.FeatureDim))
        self.clf = Sequential(Linear(1024, 256), Tanh(), Linear(256, self.conf.CLASSNUM))
        # 损失函数定义在模型类的内部
        self.KLloss_c = KLDivLoss(reduction="batchmean", log_target=True)
        self.KLloss_t = KLDivLoss(reduction="batchmean", log_target=True)
        self.CEloss_clf = CrossEntropyLoss()

    def forward(self, inputs, mode="Train"):
        """
        :param input: [{input_ids:[], ..., mask...}, {} ...]
        :param mode: train 返回三个loss， gen 返回生成的向量， test 返回分类器结果
        """

        if mode is "test":
            cls = self.bert_cls_layer(inputs)
            cls = self.linear(cls)  # shape (batch, FeatureDim)
            cls = self.dropout(cls)
            pred = self.clf(cls)
            return pred

        else:
            disentangled_features = []  # [{"h_c": h_c, "h_t": h_t}, {}, ...]
            if self.conf.fd_h_rep is False:
                h_reps = []
            for each in inputs:
                # feature disentangle, get h_c, h_t
                cls = self.bert_cls_layer(each)
                cls = self.linear(cls)  # shape (batch, FeatureDim)
                cls = self.dropout(cls)
                if self.conf.fd_h_rep is False:
                    h_reps.append(cls)
                if self.conf.loss_back_to_bert is False:
                    cls_c = cls.detach()
                    cls_t = cls.detach()
                    h_c = self.mlp_c(cls_c)
                    h_t = self.mlp_t(cls_t)
                else:
                    h_c = self.mlp_c(cls)
                    h_t = self.mlp_t(cls)
                disentangled_features.append({"h_c": h_c, "h_t": h_t})
            assert len(disentangled_features) == self.conf.CLASSNUM
            # 特征重组
            # 每一个类别的h_t都要和其它类别的h_c重组一次
            # 元素在列表中的位置就是它的label
            combined_features = [{"h_rep": [], "h_hat": []} for _ in range(self.conf.CLASSNUM)]
            for i in range(self.conf.CLASSNUM):
                for j in range(self.conf.CLASSNUM):
                    conbined_feature = torch.add(disentangled_features[i]["h_t"], disentangled_features[j]["h_c"])
                    if i == j:  # 原来的真特征
                        if self.conf.fd_h_rep is False:
                            combined_features[i]["h_rep"].append(h_reps[i])
                        else:
                            combined_features[i]["h_rep"].append(conbined_feature)
                    else:  # 不同类别的特征组合
                        combined_features[i]["h_hat"].append(conbined_feature)
            assert len(combined_features) == self.conf.CLASSNUM

            if mode is "train":
                # 再次对重组的特征进行解耦
                # disentangle_combined_features的结构为 {"h_hat_c": [h_hat_c], "h_hat_t": [h_hat_t]}
                h_hat_c = [torch.zeros((self.conf.BATCHSIZE, self.conf.FeatureDim),
                                       dtype=torch.float32, device=self.conf.DEVICE)
                           for _ in range(self.conf.CLASSNUM)]
                h_hat_t = [torch.zeros((self.conf.BATCHSIZE, self.conf.FeatureDim),
                                       dtype=torch.float32, device=self.conf.DEVICE)
                           for _ in range(self.conf.CLASSNUM)]
                for i in range(self.conf.CLASSNUM):
                    for j in range(self.conf.CLASSNUM - 1):
                        h_hat_t[i] = torch.add(h_hat_t[i], self.mlp_t(combined_features[i]["h_hat"][j]))
                        if j < i:
                            h_hat_c[j] = torch.add(h_hat_c[j], self.mlp_c(combined_features[i]["h_hat"][j]))
                        else:
                            h_hat_c[j + 1] = torch.add(h_hat_c[j + 1], self.mlp_c(combined_features[i]["h_hat"][j]))
                for i in range(self.conf.CLASSNUM):
                    h_hat_t[i] = torch.div(h_hat_t[i], (self.conf.CLASSNUM - 1))
                    h_hat_c[i] = torch.div(h_hat_c[i], (self.conf.CLASSNUM - 1))
                assert len(h_hat_t) == self.conf.CLASSNUM
                assert len(h_hat_c) == self.conf.CLASSNUM
                disentangle_combined_features = {"h_hat_c": h_hat_c, "h_hat_t": h_hat_t}

                klloss_c, klloss_t, celoss_clf = 0., 0., 0.
                shuffleidx = [_ for _ in range(self.conf.CLASSNUM)]
                random.shuffle(shuffleidx)
                for i in range(self.conf.CLASSNUM):
                    klloss_c += self.KLloss_c(F.log_softmax(disentangled_features[shuffleidx[i]]["h_c"]),
                                              F.log_softmax(disentangle_combined_features["h_hat_c"][shuffleidx[i]],
                                                            dim=1))
                    klloss_t += self.KLloss_t(F.log_softmax(disentangled_features[shuffleidx[i]]["h_t"]),
                                              F.log_softmax(disentangle_combined_features["h_hat_t"][shuffleidx[i]],
                                                            dim=1))
                    celoss_clf += self.CEloss_clf(self.clf(combined_features[shuffleidx[i]]["h_rep"][0]),
                                                  # h_rep输入数据的特征向量，每个类别只有一个
                                                  torch.tensor([shuffleidx[i] for _ in range(self.conf.BATCHSIZE)],
                                                               dtype=torch.long,
                                                               device=self.conf.DEVICE))
                    for h_hat in combined_features[shuffleidx[i]]["h_hat"]:  # h_rep特和解耦再重组得到特征向量，每个类别有类别数-1个
                        celoss_clf += self.CEloss_clf(self.clf(h_hat),
                                                      torch.tensor([shuffleidx[i] for _ in range(self.conf.BATCHSIZE)],
                                                                   dtype=torch.long,
                                                                   device=self.conf.DEVICE))
                klloss_c /= self.conf.CLASSNUM
                klloss_t /= self.conf.CLASSNUM
                celoss_clf /= self.conf.CLASSNUM ** 2

                return klloss_c, klloss_t, celoss_clf, combined_features

            elif mode is "clf_only":  # 不进行gan训练，只训练生成器中的clf看看clf和采样方法的效果，仅用于开发过程
                celoss_clf = 0.
                for i in range(self.conf.CLASSNUM):
                    celoss_clf += self.CEloss_clf(self.clf(combined_features[i]["h_rep"][0]),
                                                  torch.tensor([i for _ in range(self.conf.BATCHSIZE)],
                                                               dtype=torch.long,
                                                               device=self.conf.DEVICE))
                celoss_clf /= self.conf.CLASSNUM
                return celoss_clf


# ---------------------------------Binary Cross Entropy---------------------------------------------
class Discrimitor(Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        # self.clf = Sequential(Linear(self.conf.FeatureDim, 1024), Tanh(), Dropout(),
        #                       Linear(1024, 512), Tanh(), Dropout(),
        #                       Linear(512, 256), Tanh(), Dropout(),
        #                       Linear(256, 1), Sigmoid())
        self.clf = Sequential(Linear(self.conf.FeatureDim, 1024), GELU(), Dropout(),
                              Linear(1024, 512), GELU(), Dropout(),
                              Linear(512, 256), GELU(), Dropout(),
                              Linear(256, 1), Sigmoid())
        self.bceloss = BCELoss()

    def forward(self, combined_features, trainmode="gen"):
        # trainmode取值"gen"或"dis"
        # 取值为dis时用于训练鉴别器，对于传入的特征，假特征标签Fake为0,真特征True为1
        # 取值为gen时用于训练生成器，对于传入的特征，真特征忽略， 假特征打上真标签，希望loss回传时使得生成器更能生成优质的假特征
        celoss_d = 0.
        h_rep, h_hat = [], []
        if trainmode is "dis":
            for i in range(self.conf.CLASSNUM):
                h_rep += combined_features[i]["h_rep"]
                h_hat += combined_features[i]["h_hat"]
            for i in range(self.conf.CLASSNUM * (self.conf.CLASSNUM - 1)):
                celoss_d += self.bceloss(self.clf(h_rep[i % self.conf.CLASSNUM]).squeeze(),  # h_rep重复采样，确保真实特征与合成特征数量一致
                                         torch.tensor([1 for _ in range(self.conf.BATCHSIZE)], dtype=torch.float,
                                                      device=self.conf.DEVICE))
                celoss_d += self.bceloss(self.clf(h_hat[i]).squeeze(),
                                         torch.tensor([0 for _ in range(self.conf.BATCHSIZE)],
                                                      dtype=torch.float,
                                                      device=self.conf.DEVICE))

            # 这里实际就等效于d_loss = d_loss_real + d_loss_fake
            celoss_d /= (self.conf.CLASSNUM * (self.conf.CLASSNUM - 1) * 2)
        elif trainmode is "gen":
            for i in range(self.conf.CLASSNUM):
                for h_hat in combined_features[i]["h_hat"]:  # 假特征打上真标签
                    celoss_d += self.bceloss(self.clf(h_hat).squeeze(),
                                             torch.tensor([1 for _ in range(self.conf.BATCHSIZE)],
                                                          dtype=torch.float,
                                                          device=self.conf.DEVICE))

            celoss_d /= (self.conf.CLASSNUM * (self.conf.CLASSNUM - 1))  # 这里实际就等效于d_loss = d_loss_real + d_loss_fake
        return celoss_d

# ------------------Cross Entropy -----------------------------------------------------
# class Discrimitor(Module):
#     def __init__(self, conf):
#         super().__init__()
#         self.conf = conf
#         self.clf = Sequential(Linear(self.conf.FeatureDim, 1024), Tanh(), Dropout(),
#                               Linear(1024, 512), Tanh(), Dropout(),
#                               Linear(512, 256), Tanh(), Dropout(),
#                               Linear(256, 2))
#         self.MSEloss = CrossEntropyLoss()
#         # self.MSEloss = MSELoss()
#
#     def forward(self, combined_features, trainmode="gen"):
#         # trainmode取值"gen"或"dis"
#         # 取值为dis时用于训练鉴别器，对于传入的特征，假特征标签Fake为0,真特征True为1
#         # 取值为gen时用于训练生成器，对于传入的特征，真特征忽略， 假特征打上真标签，希望loss回传时使得生成器更能生成优质的假特征
#         celoss_d = 0.
#         h_rep, h_hat = [], []
#         if trainmode is "dis":
#             for i in range(self.conf.CLASSNUM):
#                 h_rep += combined_features[i]["h_rep"]
#                 h_hat += combined_features[i]["h_hat"]
#             for i in range(self.conf.CLASSNUM * (self.conf.CLASSNUM - 1)):
#                 celoss_d += self.MSEloss(self.clf(h_rep[i % self.conf.CLASSNUM]),  # h_rep重复采样，确保真实特征与合成特征数量一致
#                                          torch.tensor([1 for _ in range(self.conf.BATCHSIZE)], dtype=torch.long,
#                                                       device=self.conf.DEVICE))
#                 celoss_d += self.MSEloss(self.clf(h_hat[i]),
#                                          torch.tensor([0 for _ in range(self.conf.BATCHSIZE)],
#                                                       dtype=torch.long,
#                                                       device=self.conf.DEVICE))
#
#             # 这里实际就等效于d_loss = d_loss_real + d_loss_fake
#             celoss_d /= (self.conf.CLASSNUM * (self.conf.CLASSNUM - 1) * 2)
#         elif trainmode is "gen":
#             for i in range(self.conf.CLASSNUM):
#                 for h_hat in combined_features[i]["h_hat"]:  # 假特征打上真标签
#                     celoss_d += self.MSEloss(self.clf(h_hat), torch.tensor([1 for _ in range(self.conf.BATCHSIZE)],
#                                                                            dtype=torch.long,
#                                                                            device=self.conf.DEVICE))
#
#             celoss_d /= (self.conf.CLASSNUM * (self.conf.CLASSNUM - 1))  # 这里实际就等效于d_loss = d_loss_real + d_loss_fake
#         return celoss_d
