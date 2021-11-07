from config import *
from model_util import *
import torch
torch.manual_seed(42)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def train_epoch(model, optimizer):
    model.train()
    losses = 0

    train_dataset = Papago_Dataset('./dataset/train_parallel.csv')
    train_len = int(len(train_dataset) * 0.8)
    val_len = len(train_dataset) - train_len
    train_iter, _ = random_split(train_dataset, [train_len, val_len])
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0

    train_dataset = Papago_Dataset('./dataset/train_parallel.csv')
    train_len = int(len(train_dataset) * 0.8)
    val_len = len(train_dataset) - train_len
    _, val_iter = random_split(train_dataset, [train_len, val_len])
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn, drop_last=True)
    with torch.no_grad():
        for src, tgt in val_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
    return losses / len(val_dataloader)





