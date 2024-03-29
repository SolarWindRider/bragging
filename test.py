import numpy as np

from utils import MyDataset, SimpleBertModel
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
    # args = parser.parse_args()
    # conf.MODLENAME = args.model_name
    # conf.LM = args.language_model
    conf.CLASSNUM = 2
    # device = args.device
    # tokenizer = AutoTokenizer.from_pretrained(conf.LM)
    # model = MyModel(tokenizer.vocab_size, conf).load_state_dict(torch.load(f"./models/{conf.MODLENAME}"))

    device = conf.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(conf.LM)
    model = SimpleBertModel(tokenizer.vocab_size, conf)
    # model.load_state_dict(torch.load(f"./models/nrc/{conf.MODLENAME}.pt", map_location=device))
    # model.load_state_dict(torch.load(f"./models/sampling/40epochs_finished_sampling0_noseed_weight.pt", map_location=device))
    # model.load_state_dict(torch.load(f"./models/weighted/40epochs_finished_nrc_7class.pt", map_location=device))
    model.load_state_dict(torch.load(f"./models/sampling/50epochs_finished_xxx.pt", map_location=device))
    model.to(device)

    dataset = MyDataset(tokenizer, conf, False)
    train_loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
    )

    model.eval()
    with torch.no_grad():
        pred, truth = [], []
        for inputs, labels in tqdm(train_loader):
            for k in inputs.keys():
                inputs[k] = inputs[k].to(device).squeeze()
            truth += list(labels.numpy())
            # print(inputs["input_ids"].shape)
            outputs = model(inputs)
            outputs = outputs.argmax(dim=1)
            pred += list(outputs.cpu().numpy())
    for average in ["micro", "macro"]:
        print(conf.MODLENAME)
        print(average)
        print(
            f"P: {precision_score(truth, pred, average=average):6.3%}, R: {recall_score(truth, pred, average=average):6.3%}, F1: {f1_score(truth, pred, average=average):6.3%}")
