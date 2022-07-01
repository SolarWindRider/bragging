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
    # parser.add_argument("-lm", "--languagemodel", help="pretrained language model name", type=str)
    # parser.add_argument("-n", "--model_name", help="model name to save", type=str)
    # args = parser.parse_args()

    # 1. define network
    device = "cuda:2"
    tokenizer = AutoTokenizer.from_pretrained(conf.LM)
    model = MyModel(tokenizer.vocab_size, conf).to(device)
    # DistributedDataParallel

    # 2. define dataloader
    dataset = MyDataset(tokenizer, conf)
    # DistributedSampler
    # single Machine with 3 GPUs
    train_loader = DataLoader(
        dataset,
        batch_size=conf.BATCHSIZE,
        num_workers=4,
        pin_memory=True,
    )

    # 3. define loss and optimizer
    weight = torch.tensor(conf.BIN_CLASS_WEIGHT, device=device)
    if conf.CLASSNUM == 7:
        weight = torch.tensor(conf.MULTI_CLASS_WEIGHT, device=device)
    loss_fn = CrossEntropyLoss(weight=weight, )

    optimizer = SGD([
        {'params': model.linear.parameters()},
        {'params': model.lm.parameters(), 'lr': conf.LMLR}
    ], lr=conf.LINEARLR, momentum=0.9)

    # 4. start to train
    model.train()
    for ep in range(conf.EPOCHS):
        # set sampler

        # for idx, (inputs, labels) in enumerate(train_loader):
        for inputs, labels in tqdm(train_loader):
            for k in inputs.keys():
                inputs[k] = inputs[k].to(device).squeeze()
            labels = labels.to(device)
            # print(inputs["input_ids"].shape)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
