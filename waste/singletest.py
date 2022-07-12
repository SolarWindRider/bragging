import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification, RobertaForSequenceClassification
from utils import MyDataset, MyModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import conf
from tqdm import tqdm
import argparse
import torch

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-n", "--model_name", help="model name to load", type=str)
    # parser.add_argument("-lm", "--language_model", help="pretrained language model", type=str)
    # parser.add_argument("-d", "--device", help="load modedl to device", type=str)
    # parser.add_argument("-avg", "--average-method", help="average method of evaluation matrix", type=str)
    # args = parser.parse_args()
    # conf.MODLENAME = args.model_name
    # conf.LM = args.language_model
    # device = args.device
    device = "cuda:2"

    tokenizer = AutoTokenizer.from_pretrained(conf.LM)
    if conf.LM in ["vinai/bertweet-base", "roberta-base"]:
        model = RobertaForSequenceClassification.from_pretrained(conf.LM, num_labels=conf.CLASSNUM).to(device)
    elif conf.LM == "bert-base-cased":
        model = BertForSequenceClassification.from_pretrained(conf.LM, num_labels=conf.CLASSNUM).to(device)

    model.load_state_dict(torch.load(f"./models/diff_lr/{conf.MODLENAME}.pt", map_location=device))
    # model.to(device)

    dataset = MyDataset(tokenizer, conf, istrain=True)
    test_loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
    )

    model.eval()
    with torch.no_grad():
        pred, truth = [], []
        for inputs, labels in tqdm(test_loader):
            for k in inputs.keys():
                inputs[k] = inputs[k].to(device).squeeze()
            truth += list(labels.numpy())
            # print(inputs["input_ids"].shape)
            outputs = model(**inputs)
            outputs = outputs.logits.argmax(dim=1)
            pred += list(outputs.cpu().numpy())
    for average in ["micro", "macro"]:
        print(conf.MODLENAME)
        print(average)
        print(
            f"P: {precision_score(truth, pred, average=average):6.3%}, R: {recall_score(truth, pred, average=average):6.3%}, F1: {f1_score(truth, pred, average=average):6.3%}")
