import pickle

from utils import MyDataset, SimpleBertModel, seed_all
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

if __name__ == '__main__':
    # 单卡单模型训练
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-dv", "--device", help="which device to load model and data", type=str)
    # parser.add_argument("-lm", "--languagemodel", help="pretrained language model name", type=str)
    # parser.add_argument("-n", "--model_name", help="model name to save", type=str)
    # parser.add_argument("-c", "--classnum", help="how many classes to classify", type=int)
    # parser.add_argument("-fp", "--filepath", help="filepath to save trained model", type=str)
    # parser.add_argument("-lf", "--linguestic_feature", help="filepath to save trained model", type=str, default=None)
    # parser.add_argument("-sd", "--random_seed", help="set random seed", type=int)
    # args = parser.parse_args()

    device = "cuda:1"
    filepath = "./models/weighted/"

    # device = args.device
    # filepath = args.filepath
    # conf.CLASSNUM = args.classnum
    # conf.LM = args.languagemodel
    # conf.MODLENAME = args.model_name
    # conf.DATAPATH = "./dataset/bragging_data.csv"
    # conf.LingFeature = args.linguestic_feature
    # conf.RANDSEED = args.random_seed

    seed_all(conf.RANDSEED)
    logging.basicConfig(filename=f'./logs/nrc_{conf.MODLENAME}_{conf.CLASSNUM}class.log', level=logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained(f"./orgmodels/{conf.LM}")
    model = SimpleBertModel(tokenizer.vocab_size, conf).to(device)

    train_set = MyDataset(tokenizer, conf, True)
    train_loader = DataLoader(
        train_set,
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

    class_map = {}
    if conf.CLASSNUM == 7:
        class_map = conf.MULTI_CLASS_MAP
    elif conf.CLASSNUM == 2:
        class_map = conf.BIN_CLASS_MAP
    weight_loss = compute_class_weight('balanced', classes=list(class_map.keys()), y=train_set.data[:, 1])
    weight = torch.tensor(weight_loss, device=device)
    loss_fn = CrossEntropyLoss(ignore_index=-100, reduction='mean')

    logging.info("training start")
    loss_train_li = []
    best_loss_train = 1e5
    for ep in range(conf.EPOCHS):
        model.train()
        logging.info(f"EPOCH: {ep}")
        loss_train = 0.
        # for idx, (inputs, labels) in enumerate(train_loader):
        for inputs, labels in tqdm(train_loader):
            for k in inputs.keys():
                inputs[k] = inputs[k].to(device).squeeze()
            labels = labels.to(device)
            outputs = model(inputs)
            loss_train_ = loss_fn(outputs, labels)
            loss_train = loss_train_.item()
            optimizer.zero_grad()
            loss_train_.backward()
            optimizer.step()
        model.eval()

        if loss_train < best_loss_train:
            best_loss_train = loss_train
            torch.save(model.state_dict(),
                       f"{filepath}/best_loss_train_{conf.MODLENAME}_{conf.CLASSNUM}class.pt")
            logging.info(f"best_loss_train model saved, loss: {best_loss_train}")

        loss_train_li.append(loss_train)
    torch.save(model.state_dict(),
               f"{filepath}/{conf.EPOCHS}epochs_finished_{conf.MODLENAME}_{conf.CLASSNUM}class.pt")
    logging.info(f"{conf.EPOCHS}epochs training finished. model saved")

    pickle.dump(loss_train_li, open(f"{conf.MODLENAME}_loss_train.pt", "wb"))


# import pickle
#
# from utils import MyDataset, SimpleBertModel, seed_all
# from transformers import AutoTokenizer
# from torch.utils.data import DataLoader
# from torch.optim import AdamW
# from torch.nn import CrossEntropyLoss
# from sklearn.utils.class_weight import compute_class_weight
# import conf
# from tqdm import tqdm
# import argparse
# import torch
# import logging
#
# if __name__ == '__main__':
#     # 单卡单模型训练
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-dv", "--device", help="which device to load model and data", type=str)
#     parser.add_argument("-lm", "--languagemodel", help="pretrained language model name", type=str)
#     parser.add_argument("-n", "--model_name", help="model name to save", type=str)
#     parser.add_argument("-c", "--classnum", help="how many classes to classify", type=int)
#     parser.add_argument("-fp", "--filepath", help="filepath to save trained model", type=str)
#     parser.add_argument("-lf", "--linguestic_feature", help="filepath to save trained model", type=str, default=None)
#     parser.add_argument("-sd", "--random_seed", help="set random seed", type=int)
#     args = parser.parse_args()
#
#     # device = "cuda:1"
#     # filepath = "./models/weighted/"
#     device = args.device
#     filepath = args.filepath
#     conf.CLASSNUM = args.classnum
#     conf.LM = args.languagemodel
#     conf.MODLENAME = args.model_name
#     conf.DATAPATH = "./dataset/bragging_data.csv"
#     conf.LingFeature = args.linguestic_feature
#     conf.RANDSEED = args.random_seed
#
#     seed_all(conf.RANDSEED)
#     logging.basicConfig(filename=f'./logs/nrc_{conf.MODLENAME}_{conf.CLASSNUM}class.log', level=logging.INFO)
#
#     tokenizer = AutoTokenizer.from_pretrained(f"./orgmodels/{conf.LM}")
#     model = SimpleBertModel(tokenizer.vocab_size, conf).to(device)
#
#     train_set = MyDataset(tokenizer, conf, True)
#     train_loader = DataLoader(
#         train_set,
#         batch_size=conf.BATCHSIZE,
#         num_workers=4,
#         pin_memory=True,
#         shuffle=True
#     )
#
#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#          'weight_decay': 0.01},
#         {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#          'weight_decay': 0.0}
#     ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=3e-6)
#
#     class_map = {}
#     if conf.CLASSNUM == 7:
#         class_map = conf.MULTI_CLASS_MAP
#     elif conf.CLASSNUM == 2:
#         class_map = conf.BIN_CLASS_MAP
#     weight_loss = compute_class_weight('balanced', classes=list(class_map.keys()), y=train_set.data[:, 1])
#     weight = torch.tensor(weight_loss, device=device)
#     loss_fn = CrossEntropyLoss(ignore_index=-100, reduction='mean')
#
#     logging.info("training start")
#     loss_train_li = []
#     best_loss_train = 1e5
#     for ep in range(conf.EPOCHS):
#         model.train()
#         logging.info(f"EPOCH: {ep}")
#         loss_train = 0.
#         # for idx, (inputs, labels) in enumerate(train_loader):
#         for inputs, labels in tqdm(train_loader):
#             for k in inputs.keys():
#                 inputs[k] = inputs[k].to(device).squeeze()
#             labels = labels.to(device)
#             outputs = model(inputs)
#             loss_train_ = loss_fn(outputs, labels)
#             loss_train = loss_train_.item()
#             optimizer.zero_grad()
#             loss_train_.backward()
#             optimizer.step()
#         model.eval()
#
#         if loss_train < best_loss_train:
#             best_loss_train = loss_train
#             torch.save(model.state_dict(),
#                        f"{filepath}/best_loss_train_{conf.MODLENAME}_{conf.CLASSNUM}class.pt")
#             logging.info(f"best_loss_train model saved, loss: {best_loss_train}")
#
#         loss_train_li.append(loss_train)
#     torch.save(model.state_dict(),
#                f"{filepath}/{conf.EPOCHS}epochs_finished_{conf.MODLENAME}_{conf.CLASSNUM}class.pt")
#     logging.info(f"{conf.EPOCHS}epochs training finished. model saved")
#
#     pickle.dump(loss_train_li, open(f"{conf.MODLENAME}_loss_train.pt", "wb"))
