import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

MULTI_CLASS_MAP = {
    "not": 0,  # 2838
    "achievement": 1,  # 166
    "action": 2,  # 127
    "feeling": 3,  # 39
    "trait": 4,  # 91
    "possession": 5,  # 58
    "affiliation": 6,  # 63
}

df = pd.read_csv("./dataset/bragging_data.csv", encoding="utf-8")

test = df[df["sampling"] == "random"]
test_set = test[["text", "label"]].to_numpy()
pickle.dump(test_set, open("./dataset/test_set.pt", "wb"))

data__ = df[df["sampling"] == "keyword"]
data = data__[["text", "label"]].to_numpy()
x_train, x_dev, y_train, y_dev = train_test_split(data[:, 0], data[:, 1], test_size=0.2)
train_set = np.vstack((x_train, y_train))
pickle.dump(train_set.transpose(), open("./dataset/train_set.pt", "wb"))
dev_set = np.vstack((x_dev, y_dev))
pickle.dump(dev_set.transpose(), open("./dataset/dev_set.pt", "wb"))

print("pauss")
pass
