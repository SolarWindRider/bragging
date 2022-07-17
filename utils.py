from nrclex import NRCLex
from torch.utils.data import Dataset
from torch.nn import Module, Linear, Dropout, ReLU, Parameter, LayerNorm
import torch
from transformers import AutoModelForMaskedLM
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle


# ----------------------------没有验证集----------------------------------------
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


# --------------------------bragging原文的模型复现------------------------------------------------------
class SimpleBertModel(Module):
    def __init__(self, vocab_size, conf):
        super().__init__()
        self.embedding_lm = AutoModelForMaskedLM.from_pretrained(conf.LM, output_hidden_states=True)
        self.attn_gate = AttnGating(conf)
        self.lm = AutoModelForMaskedLM.from_pretrained(conf.LM)
        if conf.LM == "vinai/bertweet-base":
            vocab_size += 1
        self.linear = Linear(vocab_size, conf.CLASSNUM)
        self.dropout = Dropout(conf.DROPOUT)

    def forward(self, inputs):
        _ = self.embedding_lm(input_ids=inputs["input_ids"],
                              token_type_ids=inputs["token_type_ids"],
                              attention_mask=inputs["attention_mask"])
        roberta_embed = _[1][0]
        combine_embed = self.attn_gate(roberta_embed, inputs["input_feature"])
        lm_output = self.lm(input_ids=None, token_type_ids=inputs["token_type_ids"],
                            attention_mask=inputs["attention_mask"], inputs_embeds=combine_embed)
        cls = lm_output.logits[:, 0, :]
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
    sns.lineplot(x='epoch', y='loss_dev', data=df, lw=3)
    plt.legend(labels=['loss_train'], facecolor='white')
    plt.show()
