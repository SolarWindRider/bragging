import numpy as np
import pandas as pd
from utils import MyDataset, SimpleBertModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import conf
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
import seaborn as sns

conf.LMoutput = True
device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained(conf.LM)
model = SimpleBertModel(tokenizer.vocab_size, conf)
model.load_state_dict(torch.load(f"./models/nrc/{conf.MODLENAME}.pt", map_location=device))
model.to(device)

dataset = MyDataset(tokenizer, conf, False)
train_loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    pin_memory=True,
)

model.eval()
with torch.no_grad():
    emb, truth = np.ones(shape=(1, 64001), dtype=np.float32), np.ones(shape=(1,), dtype=np.int64)
    for inputs, labels in tqdm(train_loader):
        for k in inputs.keys():
            inputs[k] = inputs[k].to(device).squeeze()
        truth = np.concatenate((truth, labels.cpu().numpy().squeeze()), axis=0)
        outputs = model(inputs)
        emb = np.concatenate((emb, outputs.cpu().numpy().squeeze()), axis=0)
emb = emb[1:, :]
truth = np.expand_dims(truth[1:], axis=1)
print("t-SNE start")
ts = TSNE(n_components=2, init="pca", random_state=conf.RANDSEED, learning_rate="auto")
emb = ts.fit_transform(emb)
print("t-SNE finished")

label_map = dict(zip(range(len(conf.MULTI_CLASS_MAP.keys())), list(conf.MULTI_CLASS_MAP.keys())))
label_map[0] = "non-bragging"
data = np.concatenate((emb, truth), axis=1)
df = pd.DataFrame(data, columns=["dim1", "dim2", "labels"])
for _ in label_map.keys():  # DataFrame按条件筛选赋值
    df.loc[df["labels"] == _, "labels"] = label_map[_]

plt.figure(figsize=(16, 10))
plt.title("with non-bragging")
sns.scatterplot(data=df, x="dim1", y="dim2", hue="labels", palette="Paired")
fig1 = plt.gcf()
fig1.savefig("./with non-bragging.png")
fig1.clear()

plt.figure(figsize=(16, 10))
plt.title("without non-bragging")
sns.scatterplot(data=df[df["labels"] != "non-bragging"], x="dim1", y="dim2", hue="labels", palette="Paired")
fig2 = plt.gcf()
fig2.savefig("./without non-bragging.png")
fig2.clear()
