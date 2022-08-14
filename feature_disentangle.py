from utils import GanTrainDataLoader, Generator, GanData, Discrimitor, seed_all
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from torch.nn import Linear
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
import conf
from tqdm import tqdm
import argparse
import torch
import logging
from math import ceil
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # 单卡单模型训练
    parser = argparse.ArgumentParser()
    parser.add_argument("-dv", "--device", help="which device to load model and data", type=str)
    parser.add_argument("-lm", "--languagemodel", help="pretrained language model name", type=str,
                        default="vinai/bertweet-base")
    parser.add_argument("-n", "--model_name", help="model name to save", type=str)
    parser.add_argument("-ep", "--epoches", type=int, default=50)
    parser.add_argument("-clm", "--classnum", help="how many classes to classify", type=int, default=7)
    parser.add_argument("-fp", "--filepath", help="filepath to save trained model", type=str, default="./models/gan/")
    parser.add_argument("-lf", "--linguestic_feature", help="filepath to save trained model", type=str, default=None)
    parser.add_argument("-sd", "--random_seed", help="set random seed", type=int, default=3407)
    parser.add_argument("-h_rep", "--fd_h_rep", help="whether feature recombine h_i", type=bool)
    parser.add_argument("-loss2bert", "--loss_back_to_bert", help="whether loss_back_to_bert", type=bool)
    parser.add_argument("-d_lr", "--discriminitor_learning_rate", type=float, default=3e-4)
    parser.add_argument("-g_lr", "--generator_learning_rate", type=float, default=3e-6)
    parser.add_argument("-d_extra", "--extra_steps4discriminitor", type=float, default=0.1)
    parser.add_argument("-fst_d_extra", "--first_epoch_extra_steps4discriminitor", type=float, default=0.6)
    parser.add_argument("-d_clip", "--d_gradient_clip", type=bool, default=False)
    # loss_G = conf.a * klloss_c + conf.b * klloss_t + conf.c * celoss_clf + conf.d * loss_g
    parser.add_argument("-a", "--param_a", type=float, default=1.0)
    parser.add_argument("-b", "--param_b", type=float, default=1.0)
    parser.add_argument("-c", "--param_c", type=float, default=1.0)
    parser.add_argument("-d", "--param_d", type=float, default=1.0)

    parser.add_argument("-ismlpinit", "--ismlpinit", type=bool, default=False)
    parser.add_argument("-isClassShuffle", "--isClassShuffle", type=bool, default=True)
    args = parser.parse_args()

    # device = conf.DEVICE
    # filepath = "./models/gan/"
    # d_lr = 3e-4
    # g_lr = 3e-6
    # d_extra = 0.2
    # fst_d_extra = 0.6

    filepath = args.filepath
    d_lr = args.discriminitor_learning_rate
    g_lr = args.generator_learning_rate
    conf.DEVICE = args.device
    conf.CLASSNUM = args.classnum
    conf.LM = args.languagemodel
    conf.MODLENAME = args.model_name
    conf.DATAPATH = "./dataset/bragging_data.csv"
    conf.LingFeature = args.linguestic_feature
    conf.fd_h_rep = args.fd_h_rep
    conf.loss_back_to_bert = args.loss_back_to_bert
    conf.RANDSEED = args.random_seed
    d_extra = args.extra_steps4discriminitor
    fst_d_extra = args.first_epoch_extra_steps4discriminitor
    d_clip = args.d_gradient_clip
    conf.a = args.param_a
    conf.b = args.param_b
    conf.c = args.param_c
    conf.d = args.param_d
    conf.isMLPinit = args.ismlpinit
    conf.isClassShuffle = args.isClassShuffle
    conf.EPOCHS = args.epoches

    seed_all(conf.RANDSEED)
    logging.basicConfig(filename=f'./logs/{conf.MODLENAME}.log', level=logging.INFO)
    writer = SummaryWriter(f'./runs/{conf.MODLENAME}')

    tokenizer = AutoTokenizer.from_pretrained(conf.LM)
    generator = Generator(len(tokenizer), conf).to(conf.DEVICE)
    discriminitor = Discrimitor(conf).to(conf.DEVICE)

    train_loader = GanTrainDataLoader(tokenizer, conf)

    param_optimizer = list(generator.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in generator.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in generator.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    d_optimizer = AdamW(discriminitor.parameters(), lr=d_lr)
    # g_optimizer = AdamW([{"params": genertator.mlp_c.parameters(), "lr": 3e-4},
    #                      {"params": genertator.mlp_t.parameters(), "lr": 3e-4},
    #                      {"params": genertator.clf.parameters(), "lr": 3e-4},
    #                      ], lr=3e-6)

    g_optimizer = AdamW(generator.parameters(), lr=g_lr)


    def reset_grad():
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()


    print("training start")
    logging.info("training start")
    total_step = 0
    best_macroF1 = 0.
    for idx, inputs in enumerate(train_loader):
        # for inputs in tqdm(train_loader):
        generator.train()
        discriminitor.train()
        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #
        klloss_c, klloss_t, celoss_clf, combined_features = generator(inputs, mode="train")
        loss_D = discriminitor(combined_features, trainmode="dis")
        reset_grad()
        loss_D.backward(retain_graph=True)
        if d_clip is True:
            clip_grad_value_(discriminitor.parameters(), 1.0)  # 对鉴别器进行梯度裁剪
        d_optimizer.step()
        writer.add_scalar("loss_D", loss_D.item(), total_step)
        if total_step < ceil(2838 / conf.BATCHSIZE * fst_d_extra):
            total_step += 1  # 第一个epoch多训练鉴别器
            continue
        elif total_step % ceil(2838 / conf.BATCHSIZE) < ceil(2838 / conf.BATCHSIZE * d_extra):
            total_step += 1
            continue

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

        if total_step >= ceil(2838 / conf.BATCHSIZE * conf.EPOCHS * 0.6):
            if total_step % 500 == 0:  # 加载测试
                test_set = GanData(tokenizer, conf, istrain=False)
                test_loader = DataLoader(test_set, batch_size=64)

                generator.eval()
                with torch.no_grad():
                    pred, truth = [], []
                    for (inputs, labels) in test_loader:
                        _ = generator(inputs, mode='test')
                        pred_label = torch.argmax(_, dim=-1)
                        truth += list(labels.numpy())
                        pred += list(pred_label.cpu().numpy())

                for average in ["micro", "macro"]:
                    P = precision_score(truth, pred, average=average)
                    R = recall_score(truth, pred, average=average)
                    F1 = f1_score(truth, pred, average=average)
                    print(f"g_{conf.MODLENAME}_{total_step}_{average} [P: {P:6.3%}, R: {R:6.3%}, F1: {F1:6.3%}]")
                    logging.info(f"g_{conf.MODLENAME}_{total_step}_{average} [P: {P:6.3%}, R: {R:6.3%}, F1: {F1:6.3%}]")
                    if average is "macro":
                        writer.add_scalar("F1", F1, total_step)
                        if F1 > best_macroF1:
                            best_macroF1 = F1
                            torch.save(generator.state_dict(), f"{filepath}/g_{conf.MODLENAME}_bestf1.pt")
                            torch.save(discriminitor.state_dict(), f"{filepath}/d_{conf.MODLENAME}_bestf1.pt")
                            print(f"best f1 model saved, step {total_step}")
                            logging.info(f"best f1 model saved, step {total_step}")

        total_step += 1

    torch.save(generator.state_dict(), f"{filepath}/g_{conf.MODLENAME}.pt")
    torch.save(discriminitor.state_dict(), f"{filepath}/d_{conf.MODLENAME}.pt")
    print(f"training finished, last step model saved")
    logging.info(f"training finished, last step model saved")
