import pickle
from utils import GanTrainDataLoader, Generator, Discrimitor, seed_all
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
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
    filepath = "./models/gan/"

    # filepath = args.filepath
    # conf.device = args.device
    # conf.CLASSNUM = args.classnum
    # conf.LM = args.languagemodel
    # conf.MODLENAME = args.model_name
    # conf.DATAPATH = "./dataset/bragging_data.csv"
    # conf.LingFeature = args.linguestic_feature
    # conf.RANDSEED = args.random_seed

    seed_all(conf.RANDSEED)
    logging.basicConfig(filename=f'./logs/{conf.MODLENAME}_{conf.CLASSNUM}class.log', level=logging.INFO)
    writer = SummaryWriter()

    tokenizer = AutoTokenizer.from_pretrained(conf.LM)
    genertator = Generator(len(tokenizer), conf).to(device)
    discriminitor = Discrimitor(conf).to(device)

    train_loader = GanTrainDataLoader(tokenizer, conf)

    param_optimizer = list(genertator.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in genertator.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in genertator.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    d_optimizer = AdamW(discriminitor.parameters(), lr=3e-6)
    g_optimizer = AdamW(genertator.parameters(), lr=3e-6)

    logging.info("training start")
    total_step = 0  # train discriminitor for 50 extra_step
    best_loss_D, loss_D, best_loss_G, loss_G = 10000., 10000., 10000., 10000.
    for ep in range(conf.EPOCHS):
        logging.info(f"EPOCH: {ep}")
        extra_d_step = 0
        for idx, inputs in enumerate(train_loader):
            # for inputs in tqdm(train_loader):
            total_step += 1
            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #
            genertator.eval()
            with torch.no_grad():
                conbined_features = genertator(inputs, istrain=False)
            discriminitor.train()
            loss_D = discriminitor(conbined_features)
            d_optimizer.zero_grad()
            loss_D.backward()
            d_optimizer.step()
            extra_d_step += 1
            if extra_d_step < 10:
                continue
            writer.add_scalar("loss_D", loss_D.item(), total_step)
            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #
            genertator.train()
            klloss_c, klloss_t, celoss_clf = genertator(inputs)
            loss_G = conf.a * klloss_c + conf.b * klloss_t + conf.c * celoss_clf
            g_optimizer.zero_grad()
            loss_G.backward()
            g_optimizer.step()
            writer.add_scalar("klloss_c", klloss_c.item(), total_step)
            writer.add_scalar("klloss_t", klloss_t.item(), total_step)
            writer.add_scalar("celoss_clf", celoss_clf.item(), total_step)
            writer.add_scalar("loss_G", loss_G.item(), total_step)

    torch.save(genertator.state_dict(), f"{filepath}/g.pt")
    torch.save(discriminitor.state_dict(), f"{filepath}/d.pt")
    logging.info(f"training finished, model saved")
