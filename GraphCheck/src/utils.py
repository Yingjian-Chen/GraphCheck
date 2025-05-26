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

    y_true = df["label"].map(label_mapping).tolist()

    y_pred = []

    for pred in df["pred"]:
        matches = re.findall(r"support|unsupport", pred.strip(), re.IGNORECASE)

        if len(matches) > 0:
            pred_label = label_mapping[matches[0].lower()]
        else:
            pred_label = None

        y_pred.append(pred_label)
        
    valid_indices = [i for i in range(len(y_pred)) if y_pred[i] is not None]
    y_true = [y_true[i] for i in valid_indices]
    y_pred = [y_pred[i] for i in valid_indices]

    if not y_true:
        return None, None, None, None, None
        
    y_true = [int(label) for label in y_true]
    y_pred = [int(p) for p in y_pred]

    mACC = balanced_accuracy_score(y_true, y_pred)

    return mACC