# utils/run_all_in_pool.py
from multiprocessing import Pool
import os


def run_all_in_pool(run_func, target_dir, dataset, opts, use_multiprocessing=True):
    results = []
    parallelism = opts.cpus if opts.cpus else os.cpu_count()
    if use_multiprocessing:
        with Pool(processes=parallelism) as pool:
            results = pool.starmap(
                run_func,
                [
                    (target_dir, f"instance_{i}", *data)
                    for i, data in enumerate(dataset)
                ],
            )
    else:
        for i, data in enumerate(dataset):
            results.append(run_func(target_dir, f"instance_{i}", *data))
    return results, parallelism
