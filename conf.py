DATAPATH = "./dataset/balanced.csv"
# DATAPATH = "./dataset/bragging_data.csv"
# DEVICE = "cpu" # DDP不需要这个
BATCHSIZE = 32
EPOCHS = 60
LMLR = 3e-6
LINEARLR = 1e-3
DROPOUT = 0.2
MAXLENGTH = 50
# LM = "bert-base-cased"
LM = "vinai/bertweet-base"
# MODLENAME = "best_loss_bertweet_7class"
MODLENAME = "60epochs_finished_bertweet_7class"
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
