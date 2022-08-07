from utils import SimpleBertModel, seed_all, GanTrainDataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
import conf
from tqdm import tqdm
import argparse
import torch
import logging
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # 单卡单模型训练
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-dv", "--device", help="which device to load model and data", type=str)
    # parser.add_argument("-lm", "--languagemodel", help="pretrained language model name", type=str)
    # parser.add_argument("-n", "--model_name", help="model name to save", type=str)
    # parser.add_argument("-c", "--classnum", help="how many classes to classify", type=int, default=7)
    # parser.add_argument("-fp", "--filepath", help="filepath to save trained model", type=str)
    # parser.add_argument("-lf", "--linguestic_feature", help="filepath to save trained model", type=str, default=None)
    # parser.add_argument("-sd", "--random_seed", help="set random seed", type=int)
    # args = parser.parse_args()

    device = "cuda:0"
    filepath = "./models/sampling/"

    # filepath = args.filepath
    # conf.DEVICE = args.device
    # conf.CLASSNUM = args.classnum
    # conf.LM = args.languagemodel
    # conf.MODLENAME = args.model_name
    # conf.DATAPATH = "./dataset/bragging_data.csv"
    # conf.LingFeature = args.linguestic_feature
    # conf.RANDSEED = args.random_seed

    seed_all(conf.RANDSEED)
    writer = SummaryWriter(f'./runs/{conf.MODLENAME}')
    logging.basicConfig(filename=f'./logs/sampling_{conf.MODLENAME}_{conf.CLASSNUM}class.log', level=logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained(conf.LM)
    model = SimpleBertModel(tokenizer.vocab_size, conf).to(conf.DEVICE)

    train_loader = GanTrainDataLoader(tokenizer, conf)

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

    logging.info("training start")
    best_loss_train = 1e5
    total_step = 0

    model.train()
    # for idx, inputs in enumerate(train_loader):
    for inputs in tqdm(train_loader):
        loss_train = 0.
        # shuffleidx = [_ for _ in range(conf.CLASSNUM)]
        # for _ in range(conf.CLASSNUM):
        #     for k in inputs[shuffleidx[_]].keys():
        #         inputs[shuffleidx[_]][k] = inputs[shuffleidx[_]][k].to(conf.DEVICE).squeeze()
        #     labels = torch.tensor([shuffleidx[_] for i in range(conf.BATCHSIZE)],
        #                           dtype=torch.long,
        #                           device=conf.DEVICE)
        #     outputs = model(inputs[shuffleidx[_]])

        for _ in range(conf.CLASSNUM):
            for k in inputs[_].keys():
                inputs[_][k] = inputs[_][k].to(conf.DEVICE).squeeze()
            labels = torch.tensor([_ for i in range(conf.BATCHSIZE)],
                                  dtype=torch.long,
                                  device=conf.DEVICE)
            outputs = model(inputs[_])

            loss_train_ = loss_fn(outputs, labels)
            loss_train = loss_train_.item()
            optimizer.zero_grad()
            loss_train_.backward()
            optimizer.step()
            loss_train += loss_train_

        loss_train /= conf.CLASSNUM
        writer.add_scalar("loss_train", loss_train, total_step)
        total_step += 1
        if total_step % (conf.BATCHSIZE * conf.EPOCHS) == 0:
            if loss_train < best_loss_train:
                best_loss_train = loss_train
                torch.save(model.state_dict(),
                           f"{filepath}/best_loss_train_{conf.MODLENAME}.pt")
                logging.info(f"best_loss_train model saved, loss: {best_loss_train}")

    torch.save(model.state_dict(),
               f"{filepath}/{total_step}steps_finished_{conf.MODLENAME}_{conf.CLASSNUM}class.pt")
    logging.info(f"training finished. model saved")
