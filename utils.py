from torch.utils.data import Dataset
from torch.nn import Module, Linear
from transformers import AutoModelForMaskedLM
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, tokenizer, conf, istrain=True):
        """
        :param tokenizer: 传入已经实例化完成的tokenizer
        :param istrain: bool 是否载入训练数据
        :param numclass: int 分类任务数
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
        inputs = self.tokenizer(self.data[item][0], return_tensors="pt",
                                max_length=self.conf.MAXLENGTH, padding="max_length",
                                truncation=True)
        label = self.label_map[self.data[item][1]]
        return inputs, label


class MyModel(Module):
    def __init__(self, vocab_size, conf):
        super().__init__()
        self.lm = AutoModelForMaskedLM.from_pretrained(conf.LM)
        if conf.LM == "vinai/bertweet-base":
            vocab_size += 1
        self.linear = Linear(vocab_size, conf.CLASSNUM)

    def forward(self, inputs):
        lm_output = self.lm(**inputs)
        cls = lm_output.logits[:, 0, :]
        linear_output = self.linear(cls)
        return linear_output
