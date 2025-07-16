import argparse
import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import run_all_in_pool
from utils.data_utils import check_extension, save_dataset, load_dataset
from problem.problem_NCO import NCODataset, NCOProblem
from rainforcement_baseline import GeneticBaseline


def solve_genetic(
    directory, name, services_data, nodes_data, opts, disable_cache=False
):
    """Solve NCO using GeneticBaseline."""
    problem_filename = os.path.join(directory, f"{name}.genetic.pkl")

    try:
        if os.path.isfile(problem_filename) and not disable_cache:
            (cost, assignments, duration) = load_dataset(problem_filename)
        else:
            start = time.time()
            baseline = GeneticBaseline(NCOProblem(), opts)
            cost, _ = baseline.eval(
                services_data.to(opts.device), nodes_data.to(opts.device), None
            )
            assignments = None  # GeneticBaseline does not return assignments directly
            duration = time.time() - start
            save_dataset((cost.item(), assignments, duration), problem_filename)

        return cost.item(), assignments, duration

    except Exception as e:
        print(f"Exception occurred: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method",
        help="Name of the method to evaluate, currently only 'genetic' supported",
    )
    parser.add_argument(
        "datasets", nargs="+", help="Filename of the dataset(s) to evaluate"
    )
    parser.add_argument("-f", action="store_true", help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument(
        "--cpus", type=int, help="Number of CPUs to use, defaults to all cores"
    )
    parser.add_argument("--disable_cache", action="store_true", help="Disable caching")
    parser.add_argument(
        "--max_calc_batch_size", type=int, default=32, help="Size for subbatches"
    )
    parser.add_argument(
        "--progress_bar_mininterval", type=float, default=0.1, help="Minimum interval"
    )
    parser.add_argument("-n", type=int, help="Number of instances to process")
    parser.add_argument("--offset", type=int, help="Offset where to start processing")
    parser.add_argument(
        "--results_dir", default="results", help="Name of results directory"
    )
    parser.add_argument(
        "--json_dir",
        default="data/processed",
        help="Directory of JSON data for NCODataset",
    )

    opts = parser.parse_args()

    # Add NCO-specific options
    opts.device = "cuda" if torch.cuda.is_available() else "cpu"
    opts.population_size = 50
    opts.generations = 100
    opts.mutation_rate = 0.1
    opts.crossover_rate = 0.8

    assert (
        opts.o is None or len(opts.datasets) == 1
    ), "Cannot specify result filename with more than one dataset"

    for dataset_path in opts.datasets:
        dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])

        if opts.o is None:
            results_dir = os.path.join(opts.results_dir, dataset_basename)
            os.makedirs(results_dir, exist_ok=True)
            out_file = os.path.join(
                results_dir,
                f"{dataset_basename}offs{opts.offset if opts.offset else ''}n{opts.n if opts.n else ''}{opts.method}{ext}",
            )
        else:
            out_file = opts.o

        assert opts.f or not os.path.isfile(
            out_file
        ), "File already exists! Try running with -f option to overwrite."

        if opts.method == "genetic":
            # Load NCO dataset
            nco_dataset = NCODataset(
                json_dir=opts.json_dir, num_samples=opts.n if opts.n else 1000
            )
            dataset = [
                (services_data, nodes_data) for services_data, nodes_data in nco_dataset
            ]

            def run_func(args):
                return solve_genetic(*args, opts, disable_cache=opts.disable_cache)

            results, parallelism = run_all_in_pool(
                run_func, results_dir, dataset, opts, use_multiprocessing=True
            )

            costs, assignments, durations = zip(*[r for r in results if r is not None])
            costs, durations = np.array(costs), np.array(durations)

            print(
                "Validation average cost: {:.3f} +- {:.3f}".format(
                    costs.mean(), np.std(costs)
                )
            )
            print(
                "Average duration: {:.3f}s +- {:.3f}".format(
                    durations.mean() / parallelism, np.std(durations)
                )
            )
            print("Total duration: {}s".format(np.sum(durations) / parallelism))

            save_dataset(list(zip(costs, assignments, durations)), out_file)
        else:
            assert False, f"Unknown method: {opts.method}"
