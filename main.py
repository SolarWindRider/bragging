from utils import GanTrainDataLoader, Generator, GanData, seed_all
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
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
    # parser.add_argument("-ns", "--native_sampling", help="set native sampling rate", type=float)
    # args = parser.parse_args()
    #
    device = conf.DEVICE
    filepath = "./models/gan/"
    #
    # filepath = args.filepath
    # conf.device = args.device
    # conf.CLASSNUM = args.classnum
    # conf.LM = args.languagemodel
    # conf.MODLENAME = args.model_name
    # conf.DATAPATH = "./dataset/bragging_data.csv"
    # conf.LingFeature = args.linguestic_feature
    # conf.RANDSEED = args.random_seed

    seed_all(conf.RANDSEED)
    logging.basicConfig(filename=f'./logs/{conf.MODLENAME}_clfonly_{conf.Nsampling}.log', level=logging.INFO)
    writer = SummaryWriter()

    tokenizer = AutoTokenizer.from_pretrained(conf.LM)
    genertator = Generator(len(tokenizer), conf).to(conf.DEVICE)

    train_loader = GanTrainDataLoader(tokenizer, conf)

    param_optimizer = list(genertator.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in genertator.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in genertator.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    g_optimizer = AdamW(genertator.parameters(), lr=3e-6)

    logging.info("training start")
    total_step = 0
    celoss_clf = 0.
    for idx, inputs in enumerate(train_loader):
    # for inputs in tqdm(train_loader):
        total_step += 1
        celoss_clf = genertator(inputs, "clf_only")
        g_optimizer.zero_grad()
        celoss_clf.backward()
        g_optimizer.step()
        writer.add_scalar("celoss_clf", celoss_clf.item(), total_step)
        if (total_step - 1) % (conf.BATCHSIZE * conf.EPOCHS) == 0:
            torch.save(genertator.state_dict(), f"{filepath}/g_clfonly_{conf.EPOCHS}epochs_{conf.Nsampling}.pt")
            print(f"model saved, step:{total_step}")

            test_set = GanData(tokenizer, conf, istrain=False)
            test_loader = DataLoader(test_set, batch_size=conf.BATCHSIZE)

            genertator.eval()
            with torch.no_grad():
                pred, truth = [], []
                # for idx, (inputs, label) in enumerate(test_loader):
                for (inputs, labels) in tqdm(test_loader):
                    _ = genertator(inputs, mode='test')
                    pred_label = torch.argmax(_, dim=-1)
                    truth += list(labels.numpy())
                    pred += list(pred_label.cpu().numpy())

            print(conf.MODLENAME)
            for average in ["micro", "macro"]:
                print(average)
                print(
                    f"P: {precision_score(truth, pred, average=average):6.3%}, R: {recall_score(truth, pred, average=average):6.3%}, F1: {f1_score(truth, pred, average=average):6.3%}")
                logging.info(average)
                logging.info(
                    f"P: {precision_score(truth, pred, average=average):6.3%}, R: {recall_score(truth, pred, average=average):6.3%}, F1: {f1_score(truth, pred, average=average):6.3%}")
    torch.save(genertator.state_dict(), f"{filepath}/g_clfonly_{total_step}step_{conf.Nsampling}.pt")
    logging.info(f"training finished, model saved")
