from utils import MyDataset
from transformers import AutoTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, \
    get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import SGD
import conf
from tqdm import tqdm
import argparse
import torch
import logging

if __name__ == '__main__':
    # 单卡单模型训练
    parser = argparse.ArgumentParser()
    parser.add_argument("-dv", "--device", help="which device to load model and data", type=str)
    parser.add_argument("-lm", "--languagemodel", help="pretrained language model name", type=str)
    parser.add_argument("-n", "--model_name", help="model name to save", type=str)
    parser.add_argument("-c", "--classnum", help="how many classes to classify", type=int)
    parser.add_argument("-fp", "--filepath", help="filepath to save trained model", type=str)
    args = parser.parse_args()

    device = args.device
    filepath = args.filepath
    conf.CLASSNUM = args.classnum
    conf.LM = args.languagemodel
    conf.MODLENAME = args.model_name

    logging.basicConfig(filename=f'./logs/balanced_{conf.MODLENAME}_{conf.CLASSNUM}class.log', level=logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained(conf.LM)
    if conf.LM in ["vinai/bertweet-base", "roberta-base"]:
        model = RobertaForSequenceClassification.from_pretrained(conf.LM, num_labels=conf.CLASSNUM).to(device)
    elif conf.LM == "bert-base-cased":
        model = BertForSequenceClassification.from_pretrained(conf.LM, num_labels=conf.CLASSNUM).to(device)

    dataset = MyDataset(tokenizer, conf)

    train_loader = DataLoader(
        dataset,
        batch_size=conf.BATCHSIZE,
        num_workers=4,
        pin_memory=True,
        shuffle=True
    )

    optimizer = SGD(model.parameters(), lr=conf.LMLR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * conf.EPOCHS * len(train_loader),
        num_training_steps=conf.EPOCHS * len(train_loader)
    )
    print(model.state_dict())
    model.train()
    logging.info("training start")
    best_loss = 1e5
    for ep in range(conf.EPOCHS):
        logging.info(f"EPOCH: {ep}")
        train_loss = 0.
        for idx, (inputs, labels) in enumerate(train_loader):
            # for inputs, labels in tqdm(train_loader):
            for k in inputs.keys():
                inputs[k] = inputs[k].to(device).squeeze()
            labels = labels.to(device)
            # print(inputs["input_ids"].shape)
            outputs = model(**inputs, labels=labels)
            optimizer.zero_grad()
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()  # torch的scheduler放在epoch循环，transformers的要放在里面的循环
            train_loss = outputs.loss.item()
            if idx % 50 == 0:
                print(f"train_loss: {train_loss:.3f}")
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(),
                       f"{filepath}/best_loss_{conf.MODLENAME}_{conf.CLASSNUM}class.pt")
            logging.info(f"best loss model saved, loss: {best_loss}")

    torch.save(model.state_dict(),
               f"{filepath}/{conf.EPOCHS}epochs_finished_{conf.MODLENAME}_{conf.CLASSNUM}class.pt")
    logging.info(f"{conf.EPOCHS}epochs training finished. model saved")
