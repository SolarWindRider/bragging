from utils import GanTrainDataLoader, Generator, Discrimitor, seed_all
from transformers import AutoTokenizer
from torch.optim import AdamW
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-dv", "--device", help="which device to load model and data", type=str)
    parser.add_argument("-lm", "--languagemodel", help="pretrained language model name", type=str)
    parser.add_argument("-n", "--model_name", help="model name to save", type=str)
    parser.add_argument("-c", "--classnum", help="how many classes to classify", type=int, default=7)
    parser.add_argument("-fp", "--filepath", help="filepath to save trained model", type=str)
    parser.add_argument("-lf", "--linguestic_feature", help="filepath to save trained model", type=str, default=None)
    parser.add_argument("-sd", "--random_seed", help="set random seed", type=int)
    args = parser.parse_args()

    # device = conf.DEVICE
    # filepath = "./models/gan/"

    filepath = args.filepath
    conf.DEVICE = args.device
    conf.CLASSNUM = args.classnum
    conf.LM = args.languagemodel
    conf.MODLENAME = args.model_name
    conf.DATAPATH = "./dataset/bragging_data.csv"
    conf.LingFeature = args.linguestic_feature
    conf.RANDSEED = args.random_seed

    seed_all(conf.RANDSEED)
    logging.basicConfig(filename=f'./logs/{conf.MODLENAME}.log', level=logging.INFO)
    writer = SummaryWriter(f'./runs/{conf.MODLENAME}')

    tokenizer = AutoTokenizer.from_pretrained(f"./orgmodels/{conf.LM}")
    genertator = Generator(len(tokenizer), conf).to(conf.DEVICE)
    discriminitor = Discrimitor(conf).to(conf.DEVICE)

    train_loader = GanTrainDataLoader(tokenizer, conf)

    param_optimizer = list(genertator.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in genertator.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in genertator.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    d_optimizer = AdamW(discriminitor.parameters(), lr=3e-4)
    g_optimizer = AdamW([{"params": genertator.mlp_c.parameters(), "lr": 3e-4},
                         {"params": genertator.mlp_t.parameters(), "lr": 3e-4},
                         {"params": genertator.clf.parameters(), "lr": 3e-4},
                         ], lr=3e-6)


    # g_optimizer = AdamW(genertator.parameters(), lr=3e-6)

    def reset_grad():
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()


    logging.info("training start")
    total_step = 0
    for idx, inputs in enumerate(train_loader):
        # for inputs in tqdm(train_loader):
        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #
        klloss_c, klloss_t, celoss_clf, combined_features = genertator(inputs, mode="train")
        loss_D = discriminitor(combined_features, trainmode="dis")
        reset_grad()
        loss_D.backward(retain_graph=True)
        d_optimizer.step()
        writer.add_scalar("loss_D", loss_D.item(), total_step)
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #
        loss_g = discriminitor(combined_features, trainmode="gen")
        loss_G = conf.a * klloss_c + conf.b * klloss_t + conf.c * celoss_clf + conf.d * loss_g
        reset_grad()
        loss_G.backward()
        g_optimizer.step()
        writer.add_scalar("klloss_c", klloss_c.item(), total_step)
        writer.add_scalar("klloss_t", klloss_t.item(), total_step)
        writer.add_scalar("celoss_clf", celoss_clf.item(), total_step)
        writer.add_scalar("loss_g", loss_g.item(), total_step)
        writer.add_scalar("loss_G", loss_G.item(), total_step)
        # if total_step >= 4000 and total_step % 500 == 0:
        #     torch.save(genertator.state_dict(), f"{filepath}/g_step{total_step}.pt")
        #     torch.save(discriminitor.state_dict(), f"{filepath}/d_step{total_step}.pt")
        total_step += 1

    torch.save(genertator.state_dict(), f"{filepath}/g.pt")
    torch.save(discriminitor.state_dict(), f"{filepath}/d.pt")
    logging.info(f"training finished, model saved")
