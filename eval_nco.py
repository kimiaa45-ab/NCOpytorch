#!/usr/bin/env python
from fileinput import filename
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from problem.problem_NCO import NCODataset, NCOProblem
from utils import load_model, move_to
from train import set_decode_type
from options import get_options


def evaluate(dataset_path, opts):
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")
    problem = NCOProblem()
    model, _ = load_model(opts.model)
    model.to(opts.device)
    set_decode_type(model, opts.decode_strategy)
    model.eval()

    dataset = NCODataset(
        json_dir=opts.json_dir, num_samples=opts.val_size, filename=dataset_path
    )
    dataloader = DataLoader(
        dataset,
        batch_size=opts.eval_batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
    )

    costs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, disable=opts.no_progress_bar, ascii=True):
            services_data, nodes_data = batch
            services_data = move_to(services_data, opts.device)
            nodes_data = move_to(nodes_data, opts.device)
            if opts.decode_strategy == "bs":
                cost, _ = model(services_data, nodes_data, beam_size=opts.width)
            else:
                cost, _ = model(services_data, nodes_data)
            costs.append(cost.cpu())

    costs = torch.cat(costs, 0)
    print(f"Evaluation on {dataset_path}:")
    print(
        f"Average execution time: {costs.mean().item():.3f} ± {costs.std().item():.3f}"
    )
    return costs.mean().item()


if __name__ == "__main__":
    opts = get_options()
    dataset_path = opts.val_datasets[0] if opts.val_datasets else None
    if not dataset_path:
        raise ValueError("Please provide a dataset path for evaluation")
    evaluate(dataset_path, opts)
