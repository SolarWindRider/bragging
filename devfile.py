import pickle

from utils import MyDataset, BertClsLayer, seed_all
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight
import conf
from tqdm import tqdm
import argparse
import torch
import logging
from utils import GanTrainDataLoader, GanData
#
tokenizer = AutoTokenizer.from_pretrained(conf.LM)
trainloader = GanTrainDataLoader(tokenizer, conf)
for i in tqdm(trainloader):
    pass
# print("pause")

# a = list()
# print(a)