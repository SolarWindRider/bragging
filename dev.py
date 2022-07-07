import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification, RobertaForSequenceClassification
from utils import MyDataset, MyModel
from transformers import AutoTokenizer
import conf
import torch

tokenizer = AutoTokenizer.from_pretrained(conf.LM)
tokens = tokenizer("how are you ?", return_tensors="pt",
                        max_length=conf.MAXLENGTH, padding="max_length",
                        truncation=True)

print(tokens)