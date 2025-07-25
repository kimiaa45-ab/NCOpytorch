import math
import warnings
import torch
import numpy as np
import os
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import torch.nn.functional as F
import yaml

# لود تنظیمات از فایل YAML
config_path = "configs/config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# استخراج پارامترهای تنظیمات
num_epochs = config["model"]["num_epochs"]
num_samples = config["model"]["num_samples"]
num_layers = config["model"]["num_layers"]
gpu_hidden_dim = config["model"]["gpu_hidden_dim"]
cpu_hidden_dim = config["model"]["cpu_hidden_dim"]
device = config["model"]["device"]
charnum_s = config["model"]["charnum_s"]
charnum_n = config["model"]["charnum_n"]
charnum_se = config["model"]["charnum_se"]
charnum_ne = config["model"]["charnum_ne"]
charnum_node = config["model"]["charnum_node"]
charnum_component = config["model"]["charnum_component"]
charnum_service = config["model"]["charnum_service"]
charnum_user = config["model"]["charnum_user"]  # تعداد user nodes
charnum_helper = config["model"]["charnum_helper"]  # تعداد helper nodes
n_heads = config["model"].get("n_heads")  # پیش‌فرض 4

# تعیین دستگاه و dim
if device == "auto":
    if torch.cuda.is_available():
        device = "cuda"
        hidden_dim = gpu_hidden_dim
        batch_size = gpu_hidden_dim
        embedding_dim = gpu_hidden_dim
    else:
        device = "cpu"
        hidden_dim = cpu_hidden_dim
        batch_size = cpu_hidden_dim
        embedding_dim = cpu_hidden_dim
else:
    device = device.lower()
    hidden_dim = gpu_hidden_dim if device == "cuda" else cpu_hidden_dim


def load_problem(name):
    from problem.problem_NCO import NCODataset  # استفاده از NCOProblem

    problem = {"nco": NCODataset}.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path):
    return torch.load(
        load_path, map_location=lambda storage, loc: storage
    )  # Load on CPU


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""
    load_optimizer_state_dict = None
    print("\nLoading model from {}".format(load_path))

    load_data = torch.load(
        os.path.join(os.getcwd(), load_path), map_location=lambda storage, loc: storage
    )

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get("optimizer", None)
        load_model_state_dict = load_data.get("model", load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()
    state_dict.update(load_model_state_dict)
    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, "r") as f:
        args = json.load(f)

    # Backwards compatibility
    if "data_distribution" not in args:
        args["data_distribution"] = None
        probl, *dist = args["problem"].split("_")
        if probl == "op":
            args["problem"] = probl
            args["data_distribution"] = dist[0]

    if "knn_strat" not in args:
        args["knn_strat"] = None

    if "aggregation_graph" not in args:
        args["aggregation_graph"] = "mean"

    return args


def load_model(path, epoch=None, extra_logging=False):
    from nets.attention_decor import AttentionModel
    from nets.encoder.gnn_encoder import GNNEncoder

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == ".pt"
            )
        model_filename = os.path.join(path, "epoch-{}.pt".format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, "args.json"))
    problem = load_problem(args["problem"])

    model_class = {"attention": AttentionModel}[args.get("model", "attention")]
    assert model_class is not None, "Unknown model: {}".format(args.get("model"))
    encoder_class = {"gnn": GNNEncoder}.get(args.get("encoder", "gnn"), None)
    assert encoder_class is not None, "Unknown encoder: {}".format(args.get("encoder"))

    model = model_class(
        problem=problem,
        encoder_class=encoder_class,
        aggregation=args.get("aggregation", "mean"),
        aggregation_graph=args.get("aggregation_graph", "mean"),
        normalization=args.get("normalization", "batch"),
        learn_norm=args.get("learn_norm", True),
        track_norm=args.get("track_norm", True),
        gated=args.get("gated", True),
        tanh_clipping=args.get("tanh_clipping", 10),
        mask_inner=True,
        mask_logits=True,
        mask_graph=False,
        checkpoint_encoder=args.get("checkpoint_encoder", False),
        # shrink_size=args.get("shrink_size", 100),
        extra_logging=extra_logging,
    )

    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get("model", {})})
    model, *_ = _load_model_file(model_filename, model)
    model.eval()  # Put in eval mode

    return model, args


def parse_softmax_temperature(raw_temp):
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus
    w = len(str(len(dataset) - 1))
    offset = getattr(opts, "offset", None)
    if offset is None:
        offset = 0
    ds = dataset[offset : (offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (
        Pool
        if use_multiprocessing and num_cpus is not None and num_cpus > 1
        else ThreadPool
    )
    with pool_cls(num_cpus) as pool:
        results = list(
            tqdm(
                pool.imap(
                    func,
                    [
                        (directory, str(i + offset).zfill(w), *problem)
                        for i, problem in enumerate(ds)
                    ],
                ),
                total=len(ds),
                mininterval=opts.progress_bar_mininterval,
                ascii=True,
            )
        )

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)
    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    input = do_batch_rep(input, batch_rep)
    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        cost, mask = get_cost_func(input, pi)
        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    pis = torch.cat([F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis], 1)
    costs = torch.cat(costs, 1)
    mincosts, argmincosts = costs.min(-1)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]
    return minpis, mincosts


def get_best(sequences, cost, ids=None, batch_size=None):
    if ids is None:
        idx = cost.argmin()
        return sequences[idx : idx + 1, ...], cost[idx : idx + 1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)
    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(
        len(group_lengths) if batch_size is None else batch_size, -1, dtype=int
    )
    result[ids[all_argmin[::-1]]] = all_argmin[::-1]
    return [sequences[i] if i >= 0 else None for i in result], [
        cost[i] if i >= 0 else math.inf for i in result
    ]
