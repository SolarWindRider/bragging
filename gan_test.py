import pickle
from utils import GanData, Generator, seed_all
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import conf
from tqdm import tqdm
import argparse
import torch

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

    device = conf.DEVICE
    filepath = "./models/gan/work1"

    # filepath = args.filepath
    # conf.device = args.device
    # conf.CLASSNUM = args.classnum
    # conf.LM = args.languagemodel
    # conf.MODLENAME = args.model_name
    # conf.DATAPATH = "./dataset/bragging_data.csv"
    # conf.LingFeature = args.linguestic_feature
    # conf.RANDSEED = args.random_seed

    seed_all(conf.RANDSEED)
    tokenizer = AutoTokenizer.from_pretrained(f"./orgmodels/{conf.LM}")
    genertator = Generator(len(tokenizer), conf).to(device)
    # genertator.load_state_dict(torch.load(f"./models/gan/g_clfonly_40epochs.pt", map_location=device))
    genertator.load_state_dict(torch.load(f"./models/gan/work6/g.pt", map_location=device))

    test_set = GanData(tokenizer, conf, istrain=False)
    test_loader = DataLoader(test_set, batch_size=64)

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
