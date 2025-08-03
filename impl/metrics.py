from sklearn.metrics import f1_score, roc_auc_score
import numpy as np


def binaryf1(pred, label):
    '''
    process multi-label target
    :param pred:
    :param label:
    :return:
    '''
    pred_i = (pred > 0).astype(np.int64)
    label_i = label.reshape(pred.shape[0], -1)
    return f1_score(label_i, pred_i, average='macro')

def microf1(pred, label):
    '''
    multi-class micro-f1
    :param pred:
    :param label:
    :return:
    '''
    pred_i = np.argmax(pred, axis=1)
    return f1_score(label, pred_i, average="micro")

def auroc(pred, label):
    '''
    calculate auroc
    :param pred:
    :param label:
    :return:
    '''
    return roc_auc_score(label, pred)