RANDSEED = 3407
# DATAPATH = "./dataset/balanced.csv"
DATAPATH = "./dataset/bragging_data.csv"
DEVICE = "cuda:1"  # DDP不需要这个
LingFeature = "NRC"  # bragging原文使用的语言特征, "NRC", "LIWC", "Clusters", 默认值为None
FeatureDim = 1024
a = 1  # 生成器loss的系数
b = 1
c = 1
BATCHSIZE = 14
EPOCHS = 2
LMLR = 3e-6
LINEARLR = 1e-3
DROPOUT = 0.2
MAXLENGTH = 50
Beta = 0.001
LMoutput = False
# LM = "bert-base-cased"
LM = "vinai/bertweet-base"
MODLENAME = "gan"
# MODLENAME = "best_loss_train_bertweet-nrc_7class"
# MODLENAME = "100epochs_finished_bertweet-nrc_7class"
CLASSNUM = 7  # 2
MULTI_CLASS_MAP = {
    "not": 0,  # 2838
    "achievement": 1,  # 166
    "action": 2,  # 127
    "feeling": 3,  # 39
    "trait": 4,  # 91
    "possession": 5,  # 58
    "affiliation": 6,  # 63
}
BIN_CLASS_MAP = {
    "not": 0,
    "achievement": 1,
    "action": 1,
    "feeling": 1,
    "trait": 1,
    "possession": 1,
    "affiliation": 1,
}
