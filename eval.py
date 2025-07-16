#!/usr/bin/env python

import math
import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch

# from train import rollout_groundtruth
from utils import load_model, move_to, get_best
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
from utils.functions import parse_softmax_temperature
from problem.problem_NCO import NCODataset, NCOProblem
from nets.attention_decor import AttentionModel

import warnings

warnings.filterwarnings(
    "ignore",
    message="indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead.",
)


def eval_dataset(dataset_path, decode_strategy, width, softmax_temp, opts):
    model, model_args = load_model(opts.model)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dataset = model.problem.make_dataset(
        filename=dataset_path,
        batch_size=opts.batch_size,
        num_samples=opts.val_size,
        neighbors=model_args["neighbors"],
        knn_strat=model_args["knn_strat"],
        supervised=False,
    )

    results = _eval_dataset(
        model, dataset, decode_strategy, width, softmax_temp, opts, device
    )

    costs, tours, durations = zip(*results)
    costs, tours, durations = np.array(costs), np.array(tours), np.array(durations)

    # اگر groundtruth ندارید، این بخش را حذف کنید یا با حل‌کننده بهینه جایگزین کنید
    # gt_costs = rollout_groundtruth(model.problem, dataset, opts).cpu().numpy()
    # opt_gap = ((costs/gt_costs - 1) * 100)
    # results = zip(costs, gt_costs, tours, dataset.tour_nodes, opt_gap, durations)

    print(
        "Validation average cost: {:.3f} +- {:.3f}".format(costs.mean(), np.std(costs))
    )
    print(
        "Average duration: {:.3f}s +- {:.3f}".format(
            durations.mean(), np.std(durations)
        )
    )
    print("Total duration: {}s".format(np.sum(durations) / opts.batch_size))

    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    model_name = "_".join(
        os.path.normpath(os.path.splitext(opts.model)[0]).split(os.sep)[-2:]
    )
    results_dir = os.path.join(opts.results_dir, dataset_basename)
    os.makedirs(results_dir, exist_ok=True)

    out_file = os.path.join(
        results_dir,
        "{}-{}-{}{}-t{}-{}-{}{}".format(
            dataset_basename,
            model_name,
            decode_strategy,
            width if decode_strategy != "greedy" else "",
            softmax_temp,
            opts.offset,
            opts.offset + len(costs),
            ext,
        ),
    )

    assert opts.f or not os.path.isfile(
        out_file
    ), "File already exists! Try running with -f option to overwrite."

    save_dataset(results, out_file)

    latex_str = " & ${:.3f}\\pm{:.3f}$ & ${:.3f}$s".format(  # noqa: F541
        costs.mean(), np.std(costs), np.sum(durations) / opts.batch_size
    )
    return latex_str


def _eval_dataset(model, dataset, decode_strategy, width, softmax_temp, opts, device):
    model.to(device)
    model.eval()
    model.set_decode_type(
        "greedy" if decode_strategy in ("bs", "greedy") else "sampling",
        temp=softmax_temp,
    )
    dataloader = DataLoader(
        dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers
    )

    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar, ascii=True):
        services_data, nodes_data = move_to(batch[0], device), move_to(batch[1], device)
        start = time.time()
        with torch.no_grad():
            if decode_strategy in ("sample", "greedy"):
                if decode_strategy == "greedy":
                    assert width == 0
                    assert opts.batch_size <= opts.max_calc_batch_size
                    batch_rep = 1
                    iter_rep = 1
                elif width * opts.batch_size > opts.max_calc_batch_size:
                    assert opts.batch_size == 1
                    assert width % opts.max_calc_batch_size == 0
                    batch_rep = opts.max_calc_batch_size
                    iter_rep = width // opts.max_calc_batch_size
                else:
                    batch_rep = width
                    iter_rep = 1
                sequences, costs = model.sample_many(
                    services_data, nodes_data, batch_rep=batch_rep, iter_rep=iter_rep
                )
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
            else:
                assert decode_strategy == "bs"
                cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                    services_data,
                    nodes_data,
                    beam_size=width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size,
                )
                if sequences is None:
                    sequences = [None] * batch_size
                    costs = [math.inf] * batch_size
                else:
                    sequences, costs = get_best(
                        sequences.cpu().numpy(),
                        costs.cpu().numpy(),
                        ids.cpu().numpy() if ids is not None else None,
                        batch_size,
                    )

        duration = time.time() - start
        for seq, cost in zip(sequences, costs):
            if model.problem.NAME == "nco":
                if seq is not None:
                    seq = seq.tolist()
            else:
                assert False, "Unknown problem: {}".format(model.problem.NAME)
            results.append((cost, seq, duration))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets", nargs="+", help="Filename of the dataset(s) to evaluate"
    )
    parser.add_argument("-f", action="store_true", help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument(
        "--val_size",
        type=int,
        default=12800,
        help="Number of instances used for reporting validation performance",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset where to start in dataset (default 0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use during (baseline) evaluation",
    )
    parser.add_argument(
        "--decode_strategies",
        type=str,
        nargs="+",
        help="Beam search (bs), Sampling (sample) or Greedy (greedy)",
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        help="Sizes of beam to use for beam search or number of samples for sampling",
    )
    parser.add_argument(
        "--softmax_temperature",
        type=parse_softmax_temperature,
        default=1,
        help="Softmax temperature",
    )
    parser.add_argument("--model", type=str, help="Path to model checkpoints directory")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--no_progress_bar", action="store_true", help="Disable progress bar"
    )
    parser.add_argument(
        "--compress_mask", action="store_true", help="Compress mask into long"
    )
    parser.add_argument(
        "--max_calc_batch_size", type=int, default=10000, help="Size for subbatches"
    )
    parser.add_argument(
        "--results_dir", default="results", help="Name of results directory"
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="Use multiprocessing to parallelize over multiple GPUs",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for DataLoaders"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed to use")

    opts = parser.parse_args()
    assert opts.o is None or (
        len(opts.datasets) == 1 and len(opts.widths) <= 1
    ), "Cannot specify result filename with more than one dataset or more than one width"

    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    for decode_strategy, width in zip(opts.decode_strategies, opts.widths):
        latex_str = "{}-{}{}".format(
            opts.model, decode_strategy, width if decode_strategy != "greedy" else ""
        )
        for dataset_path in opts.datasets:
            latex_str += eval_dataset(
                dataset_path, decode_strategy, width, opts.softmax_temperature, opts
            )
        with open("results/results_latex.txt", "a") as f:
            f.write(latex_str + "\n")
