import random
import numpy as np
import pandas as pd
from utils import TsneData, Generator
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
genertator = Generator(len(tokenizer), conf).to(device)
genertator.load_state_dict(torch.load(f"./models/gan/g_p27.pt", map_location=device))

conf.RANDSEED = 39763940
test_set = TsneData(tokenizer, conf, istrain=False)
test_loader = DataLoader(test_set, batch_size=70)
genertator.eval()
with torch.no_grad():
    emb, truth = np.ones(shape=(1, 64001), dtype=np.float32), np.ones(shape=(1,), dtype=np.int64)
    for inputs, labels in tqdm(test_loader):
        for k in inputs.keys():
            inputs[k] = inputs[k].to(device).squeeze()
        truth = np.concatenate((truth, labels.cpu().numpy().squeeze()), axis=0)
        outputs = genertator(inputs, mode="test")
        emb = np.concatenate((emb, outputs.cpu().numpy().squeeze()), axis=0)
emb = emb[1:, :]
truth = np.expand_dims(truth[1:], axis=1)
print("t-SNE start")
ts = TSNE(early_exaggeration=16.0, n_components=2, init="pca", random_state=conf.RANDSEED, learning_rate="auto")
emb = ts.fit_transform(emb)
print("t-SNE finished")

label_map = dict(zip(range(len(conf.MULTI_CLASS_MAP.keys())), list(conf.MULTI_CLASS_MAP.keys())))
for _ in range(len(label_map)):
    if _ == 0:
        label_map[_] = "Not Bragging"
    else:
        label_map[_]= label_map[_].capitalize()
data = np.concatenate((emb, truth), axis=1)
df = pd.DataFrame(data, columns=["dim1", "dim2", "labels"])
for _ in label_map.keys():  # DataFrame按条件筛选赋值
    df.loc[df["labels"] == _, "labels"] = label_map[_]

plt.figure(figsize=(16, 10))
# plt.title("with no-bragging")
ax = sns.scatterplot(data=df, x="dim1", y="dim2", hue="labels", palette="Paired", s=600)

handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles[1:], labels=labels[1:], loc="upper left",
#           markerscale=3.0, fontsize=25, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
ax.legend(loc="upper left",
          markerscale=4.0, fontsize=25, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
linewidth = 4
ax.spines['bottom'].set_linewidth(linewidth)  # 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(linewidth)  # 设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(linewidth)  # 设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(linewidth)  # 设置上部坐标轴的粗细
plt.tick_params(width=linewidth, length=10.)  # 设置刻度线的粗细

plt.tight_layout()
plt.xlabel("")
plt.ylabel("")
# plt.xticks([])
# plt.yticks([])
plt.gca().axes.xaxis.set_ticklabels([])
plt.gca().axes.yaxis.set_ticklabels([])
fig1 = plt.gcf()
# fig1.savefig(f"./tsne_figs/seed-{conf.RANDSEED}-with-non-bragging.png")
fig1.savefig(f"./base-with-non-bragging.png")
fig1.clear()

# plt.figure(figsize=(16, 10))
# plt.title("without non-bragging")
# sns.scatterplot(data=df[df["labels"] != "non-bragging"], x="dim1", y="dim2", hue="labels", palette="Paired")
# fig2 = plt.gcf()
# fig2.savefig(f"./tsne_figs/seed-{conf.RANDSEED}-without-non-bragging.png")
# fig2.clear()
