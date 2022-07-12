from utils import MyDataset, MyModel
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.nn import CrossEntropyLoss
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
    model = MyModel(tokenizer.vocab_size, conf).to(device)

    dataset = MyDataset(tokenizer, conf)

    train_loader = DataLoader(
        dataset,
        batch_size=conf.BATCHSIZE,
        num_workers=4,
        pin_memory=True,
        shuffle=True
    )

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-6)

    loss_fn = CrossEntropyLoss(ignore_index=-100, reduction='mean')

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=0.1 * conf.EPOCHS * len(train_loader),
    #     num_training_steps=conf.EPOCHS * len(train_loader)
    # )
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
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()  # torch的scheduler放在epoch循环，transformers的要放在里面的循环
            train_loss = loss.item()
            if idx % 20 == 0:
                print(f"train_loss: {train_loss:.3f}")
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(),
                       f"{filepath}/best_loss_{conf.MODLENAME}_{conf.CLASSNUM}class.pt")
            logging.info(f"best loss model saved, loss: {best_loss}")

    torch.save(model.state_dict(),
               f"{filepath}/{conf.EPOCHS}epochs_finished_{conf.MODLENAME}_{conf.CLASSNUM}class.pt")
    logging.info(f"{conf.EPOCHS}epochs training finished. model saved")
