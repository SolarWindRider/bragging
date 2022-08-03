# --------------------------有验证集 ------------------------------------------------------
# class MyDataset(Dataset):
#     def __init__(self, tokenizer, conf, datatype):
#         """
#         :param tokenizer: 传入已经实例化完成的tokenizer
#         :param istrain: bool 是否载入训练数据
#         :param numclass: int 分类任务数
#         """
#         self.tokenizer = tokenizer
#         self.conf = conf
#         # 读取数据
#         self.data = pickle.load(open(f"./dataset/{datatype}_set.pt", "rb"))
#
#         if self.conf.CLASSNUM == 2:
#             self.label_map = conf.BIN_CLASS_MAP
#         elif self.conf.CLASSNUM == 7:
#             self.label_map = conf.MULTI_CLASS_MAP
#         else:
#             raise ValueError("numclass shoud be 2 or 7")
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, item):
#         inputs = self.tokenizer(self.data[item][0], return_tensors="pt",
#                                 max_length=self.conf.MAXLENGTH, padding="max_length",
#                                 truncation=True)
#         label = self.label_map[self.data[item][1]]
#         return inputs, label
# --------------------------有验证集 ------------------------------------------------------


# --------------------------生成器第一版---------------------------------------------------
# class Generator(Module):
#     def __init__(self, vocab_size, conf):
#         super().__init__()
#         self.conf = conf
#         self.bert_cls_layer = BertClsLayer(vocab_size, self.conf)
#         self.mlp_c = Sequential(Linear(self.conf.FeatureDim, 1024), Tanh(), Linear(1024, 512), Tanh(), Dropout(),
#                                 Linear(512, 1024), Tanh(), Dropout(), Linear(1024, self.conf.FeatureDim))
#         self.mlp_t = Sequential(Linear(self.conf.FeatureDim, 1024), Tanh(), Linear(1024, 512), Tanh(), Dropout(),
#                                 Linear(512, 1024), Tanh(), Dropout(), Linear(1024, self.conf.FeatureDim))
#         self.clf = Sequential(Linear(1024, 256), Tanh(), Linear(256, self.conf.CLASSNUM))
#         # 损失函数定义在模型类的内部
#         self.KLloss_c = KLDivLoss(reduction="batchmean", log_target=True)
#         self.KLloss_t = KLDivLoss(reduction="batchmean", log_target=True)
#         self.CEloss_clf = CrossEntropyLoss()
#
#     def forward(self, inputs, mode="Train"):
#         """
#         :param input: [{input_ids:[], ..., mask...}, {} ...]
#         :param mode: train 返回三个loss， gen 返回生成的向量， test 返回分类器结果
#         """
#
#         if mode is "test":
#             cls = self.bert_cls_layer(inputs)
#             h_c = self.mlp_c(cls)
#             h_t = self.mlp_t(cls)
#             pred = self.clf(torch.add(h_c, h_t))
#             return pred
#
#         else:
#             disentangled_features = []  # [{"h_c": h_c, "h_t": h_t}, {}, ...]
#             for each in inputs:
#                 # feature disentangle, get h_c, h_t
#                 cls = self.bert_cls_layer(each)
#                 h_c = self.mlp_c(cls)
#                 h_t = self.mlp_t(cls)
#                 disentangled_features.append({"h_c": h_c, "h_t": h_t})
#             assert len(disentangled_features) == self.conf.CLASSNUM
#             # 特征重组
#             # 每一个类别的h_t都要和其它类别的h_c重组一次
#             # 元素在列表中的位置就是它的label
#             combined_features = [{"h_rep": [], "h_hat": []} for i in range(self.conf.CLASSNUM)]
#             for i in range(self.conf.CLASSNUM):
#                 for j in range(self.conf.CLASSNUM):
#                     conbined_feature = torch.add(disentangled_features[i]["h_t"], disentangled_features[j]["h_c"])
#                     if i == j:  # 相同类别的特征组合
#                         combined_features[i]["h_rep"].append(conbined_feature)
#                     else:  # 不同类别的特征组合
#                         combined_features[i]["h_hat"].append(conbined_feature)
#             assert len(combined_features) == self.conf.CLASSNUM
#
#             if mode is "train":
#                 # 再次对重组的特征进行解耦
#                 # disentangle_combined_features {"h_hat_c": [h_hat_c], "h_hat_t": [h_hat_t]}
#                 h_hat_c = [torch.zeros((self.conf.BATCHSIZE, self.conf.FeatureDim),
#                                        dtype=torch.float32, device=self.conf.DEVICE)
#                            for _ in range(self.conf.CLASSNUM)]
#                 h_hat_t = [torch.zeros((self.conf.BATCHSIZE, self.conf.FeatureDim),
#                                        dtype=torch.float32, device=self.conf.DEVICE)
#                            for _ in range(self.conf.CLASSNUM)]
#                 for i in range(self.conf.CLASSNUM):
#                     for j in range(self.conf.CLASSNUM - 1):
#                         h_hat_t[i] = torch.add(h_hat_t[i], self.mlp_t(combined_features[i]["h_hat"][j]))
#                         if j < i:
#                             h_hat_c[j] = torch.add(h_hat_c[j], self.mlp_c(combined_features[i]["h_hat"][j]))
#                         else:
#                             h_hat_c[j + 1] = torch.add(h_hat_c[j + 1], self.mlp_c(combined_features[i]["h_hat"][j]))
#                 for i in range(self.conf.CLASSNUM):
#                     h_hat_t[i] = torch.div(h_hat_t[i], (self.conf.CLASSNUM - 1))
#                     h_hat_c[i] = torch.div(h_hat_c[i], (self.conf.CLASSNUM - 1))
#                 assert len(h_hat_t) == self.conf.CLASSNUM
#                 assert len(h_hat_c) == self.conf.CLASSNUM
#                 disentangle_combined_features = {"h_hat_c": h_hat_c, "h_hat_t": h_hat_t}
#
#                 klloss_c, klloss_t, celoss_clf = 0., 0., 0.
#                 for i in range(self.conf.CLASSNUM):
#                     klloss_c += self.KLloss_c(F.log_softmax(disentangled_features[i]["h_c"]),
#                                               F.log_softmax(disentangle_combined_features["h_hat_c"][i], dim=1))
#                     klloss_t += self.KLloss_t(F.log_softmax(disentangled_features[i]["h_t"]),
#                                               F.log_softmax(disentangle_combined_features["h_hat_t"][i], dim=1))
#                     celoss_clf += self.CEloss_clf(self.clf(combined_features[i]["h_rep"][0]),
#                                                   # h_rep是解耦后重新还原的向量，每个类别只有一个
#                                                   torch.tensor([i for _ in range(self.conf.BATCHSIZE)],
#                                                                dtype=torch.long,
#                                                                device=self.conf.DEVICE))
#                     for h_hat in combined_features[i]["h_hat"]:
#                         celoss_clf += self.CEloss_clf(self.clf(h_hat),
#                                                       torch.tensor([i for _ in range(self.conf.BATCHSIZE)],
#                                                                    dtype=torch.long,
#                                                                    device=self.conf.DEVICE))
#                 klloss_c /= self.conf.CLASSNUM
#                 klloss_t /= self.conf.CLASSNUM
#                 celoss_clf /= self.conf.CLASSNUM ** 2
#
#                 return klloss_c, klloss_t, celoss_clf
#
#             elif mode is "gen":
#                 return combined_features
#
#             elif mode is "clf_only":  # 不进行gan训练，只训练生成器中的clf看看clf和采样方法的效果，仅用于开发过程
#                 celoss_clf = 0.
#                 for i in range(self.conf.CLASSNUM):
#                     celoss_clf += self.CEloss_clf(self.clf(combined_features[i]["h_rep"][0]),
#                                                   torch.tensor([i for _ in range(self.conf.BATCHSIZE)],
#                                                                dtype=torch.long,
#                                                                device=self.conf.DEVICE))
#                 celoss_clf /= self.conf.CLASSNUM
#                 return celoss_clf
# --------------------------生成器第一版---------------------------------------------------
