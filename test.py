from utils import MyDataset, MyModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
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

    # device = args.device
    # tokenizer = AutoTokenizer.from_pretrained(conf.LM)
    # model = MyModel(tokenizer.vocab_size, conf).load_state_dict(torch.load(f"./models/{conf.MODLENAME}"))

    device = "cuda:2"
    tokenizer = AutoTokenizer.from_pretrained(conf.LM)
    model = MyModel(tokenizer.vocab_size, conf).load_state_dict(
        torch.load(f"./models/{conf.MODLENAME}.pt", map_location=device))

    dataset = MyDataset(tokenizer, conf, istrain=False)
    train_loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
    )

    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(train_loader):
            for k in inputs.keys():
                inputs[k] = inputs[k].to(device).squeeze()
            labels = labels.to(device)
            # print(inputs["input_ids"].shape)
            outputs = model(inputs)
            pred = outputs.argmex(-1)
