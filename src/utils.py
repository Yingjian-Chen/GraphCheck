import math
import random, os
import numpy as np
import torch
import pandas as pd
import re
from sklearn.metrics import balanced_accuracy_score

def adjust_learning_rate(param_group, LR, epoch, args):
    min_lr = 5e-6
    if epoch < args.warmup_epochs:
        lr = LR * epoch / args.warmup_epochs
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
    param_group["lr"] = lr
    return lr

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_accuracy(path):
    df = pd.read_json(path, lines=True)
    label_mapping = {"support": 1, "unsupport": 0}

    y_true = df["label"].map(lambda x: label_mapping.get(str(x).lower(), 0)).tolist()
    y_pred = []

    for pred in df["pred"]:
        try:
            matches = re.findall(r"unsupport|support", str(pred).strip(), re.IGNORECASE)
            if matches:
                pred_label = label_mapping[matches[0].lower()]
            else:
                pred_label = 0
        except:
            pred_label = 0
        y_pred.append(pred_label)

    y_true = [int(label) for label in y_true]
    y_pred = [int(p) for p in y_pred]
    bACC = balanced_accuracy_score(y_true, y_pred)
    return bACC
