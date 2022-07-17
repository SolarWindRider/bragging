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