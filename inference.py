import os
import torch
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import pandas as pd
from dataset.utils.dataset import KGDataset
from src.config import parse_args
from model import model_path
from src.ckpt import _reload_best_model
from src.utils import get_accuracy, seed_everything
from dataset.utils.collate import collate_fn
from model.graphcheck import GraphCheck

def main(args):

    seed = args.seed
    seed_everything(seed=seed)
    print(args)

    # Data loader
    dataset = KGDataset(args.dataset_name)
    test_loader = DataLoader(dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # Build Model
    args.llm_model_path = model_path[args.llm_model_name]
    model = GraphCheck(args=args)

    # Evaluating
    os.makedirs(f'{args.output_dir}/{args.project}', exist_ok=True)
    path = f'{args.output_dir}/{args.project}/{args.dataset_name}.csv'
    print(f'path: {path}')

    # Load Model Weights
    model = _reload_best_model(model, args)

    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    with open(path, "w") as f:
        for _, batch in enumerate(test_loader):
            with torch.no_grad():
                output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)

    # Evaluating
    bacc = get_accuracy(path)
    print(f'Test BAcc: {bacc}')


if __name__ == "__main__":
    args = parse_args()
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
