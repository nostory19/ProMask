import torch
import torch.nn.functional as F
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gp

def train(optimizer, model, dataloader, loss_fn, epoch, run, alpha):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    set_seed(epoch + run * 1000)
    for batch in dataloader:
        optimizer.zero_grad()
        pred, pred_sub_emb = model(*batch[:-1], id=0)
        loss_sub = loss_fn(pred_sub_emb, batch[-1])
        loss = loss_fn(pred, batch[-1]) + alpha * loss_sub
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    return sum(total_loss) / len(total_loss)


@torch.no_grad()
def test(model, dataloader, metrics, loss_fn, epoch, run, alpha):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    set_seed(epoch + run * 1000)
    preds = []
    ys = []
    preds_sub = []
    for batch in dataloader:
        pred, pred_sub_emb = model(*batch[:-1])
        preds.append(pred)
        preds_sub.append(pred_sub_emb)
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    pred_sub = torch.cat(preds_sub, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()) + alpha * metrics(pred_sub.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)+alpha*loss_fn(pred_sub, y)
