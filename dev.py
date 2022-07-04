import pandas as pd

MULTI_CLASS_MAP = {
    "not": 0,  # 2838
    "achievement": 1,  # 166
    "action": 2,  # 127
    "feeling": 3,  # 39
    "trait": 4,  # 91
    "possession": 5,  # 58
    "affiliation": 6,  # 63
}
df = pd.read_csv("./dataset/bragging_data.csv")
for k in MULTI_CLASS_MAP.keys():
    _ = df[(df["label"] == k) & (df["sampling"] == "keyword")]
    for i in range(2838 // len(_.values) - 1):
        df = pd.concat([df, _], axis=0)

df.to_csv("./dataset/balanced.csv")
