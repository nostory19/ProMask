import random
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

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
        pred, pred_sub_emb, emb_nodes = model(*batch[:-1], id=0)

        loss_sub = loss_fn(pred_sub_emb, batch[-1])
        loss = loss_fn(pred, batch[-1]) + alpha * loss_sub
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    return sum(total_loss) / len(total_loss)

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def compute_prototypes(embeddings, labels, num_classes):
    prototypes = []
    for i in range(num_classes):
        class_embeddings = embeddings[labels == i]
        prototype = np.mean(class_embeddings, axis=0)
        prototypes.append(prototype)
    return np.array(prototypes)

@torch.no_grad()
def test(model, dataloader, metrics, loss_fn, epoch, run, alpha, plt_count, repeat):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    set_seed(epoch + run * 1000)
    preds = []
    ys = []
    preds_sub = []
    emb_Nodes = []
    for batch in dataloader:
        pred, pred_sub_emb, emb_nodes = model(*batch[:-1])
        emb_Nodes.append(emb_nodes)
        preds.append(pred)
        preds_sub.append(pred_sub_emb)
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0) # (160, 6)
    pred_sub = torch.cat(preds_sub, dim=0)
    y = torch.cat(ys, dim=0)
    emb_Nodes = torch.cat(emb_Nodes, dim=0)

    labels = y.cpu().numpy()
    embeddings = emb_Nodes.cpu().numpy()
    num_classes = len(np.unique(labels))
    if plt_count > 55:

        prototypes = compute_prototypes(embeddings, labels, num_classes)
        # ============
        filtered_indices = []

        threshold = 14.0
        for class_id in range(num_classes):
            class_mask = np.array(labels) == class_id
            class_embeddings = embeddings[class_mask]
            distances = euclidean_distances(class_embeddings, prototypes[class_id].reshape(1, -1)).flatten()
            keep_idx = np.where(distances < threshold)[0]
            class_indices = np.where(class_mask)[0][keep_idx]
            filtered_indices.extend(class_indices)

        filtered_embeddings = embeddings[filtered_indices]
        filtered_labels = np.array(labels)[filtered_indices]
        all_embeddings = np.vstack([filtered_embeddings, prototypes])

        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        all_2d = tsne.fit_transform(all_embeddings)

        emb_2d = all_2d[:len(filtered_embeddings)]  # shape: (num_nodes, 2)
        prototype_2d = all_2d[len(filtered_embeddings):]  # shape: (num_classes, 2)


        plt.figure(figsize=(8, 6))
        cmap = plt.cm.get_cmap('tab20', num_classes)
        target_class = 3

        sizes = np.where(np.array(filtered_labels) == target_class, 55, 30)
        scatter = plt.scatter(
            emb_2d[:, 0], emb_2d[:, 1],
            c=filtered_labels, cmap=cmap, s=25, alpha=0.85
        )

        for i, proto in enumerate(prototype_2d):
            color = cmap(i)
            plt.scatter(
                proto[0], proto[1],
                c=[color], marker='*', s=270,
                edgecolors='black', linewidths=1.2,
                label=f'Prototype {i}'
            )


        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.tight_layout()


        save_path = os.path.join('img_test', f'plot_{repeat}_{plt_count}.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    # return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()) + alpha * metrics(pred_sub.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)+alpha*loss_fn(pred_sub, y)
