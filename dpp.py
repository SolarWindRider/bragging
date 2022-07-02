from utils import MyDataset, MyModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import conf
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import torch


def train(lm, n, ncls, batchsize=conf.BATCHSIZE):
    conf.LM = lm
    conf.MODLENAME = n
    conf.CLASSNUM = ncls
    conf.BATCHSIZE = batchsize

    # 1. define network
    tokenizer = AutoTokenizer.from_pretrained(conf.LM)
    model = MyModel(tokenizer.vocab_size, conf).to(device)
    # DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 2. define dataloader
    dataset = MyDataset(tokenizer, conf)
    # DistributedSampler
    # single Machine with 3 GPUs
    train_sampler = DistributedSampler(dataset)
    train_loader = DataLoader(
        dataset,
        batch_size=conf.BATCHSIZE,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
    )

    # 3. define loss and optimizer
    weight = torch.tensor(conf.BIN_CLASS_WEIGHT, device=device)
    if conf.CLASSNUM == 7:
        weight = torch.tensor(conf.MULTI_CLASS_WEIGHT, device=device)
    loss_fn = CrossEntropyLoss(weight=weight, )

    optimizer = SGD([
        {'params': model.module.linear.parameters()},
        {'params': model.module.lm.parameters(), 'lr': conf.LMLR}
    ], lr=conf.LINEARLR, momentum=0.9)
    if rank == 0:
        print("            =======  Training  ======= \n")

    # 4. start to train
    model.train()
    best_acc = 0.
    for ep in range(conf.EPOCHS):
        train_loss = correct = total = 0
        # set sampler
        train_loader.sampler.set_epoch(ep)

        for idx, (inputs, labels) in enumerate(train_loader):
            # for inputs, label in tqdm(train_loader):
            for k in inputs.keys():
                inputs[k] = inputs[k].to(device).squeeze()
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += labels.size(0)
            correct += torch.eq(outputs.argmax(dim=1), labels).sum().item()

            if rank == 0 and ((idx + 1) % 25 == 0 or (idx + 1) == len(train_loader)):
                acc = 100.0 * correct / total
                loss_ = train_loss / (idx + 1)
                print(
                    f"   == step: [{idx + 1:3}/{len(train_loader)}] [{ep}/{conf.EPOCHS}] | loss: {loss_:.3f} | acc: {acc:6.3f}%")
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.module.state_dict(),
                               f"./models/best_acc_{acc:6.3f}%_{conf.MODLENAME}_{conf.CLASSNUM}class.pt")
    if rank == 0:
        print("\n            =======  Training Finished  ======= \n")

        torch.save(model.module.state_dict(),
                   f"./models/{conf.EPOCHS}epochs_finished_{conf.MODLENAME}_{conf.CLASSNUM}class.pt")


if __name__ == '__main__':
    # 0. set up distributed device
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")

    train("bert-base-cased", "bert", 7)
    train("roberta-base", "roberta", 7)
    train("vinai/bertweet-base", "bertweet", 7)

    train("bert-base-cased", "bert", 2)
    train("roberta-base", "roberta", 2)
    train("vinai/bertweet-base", "bertweet", 2)
