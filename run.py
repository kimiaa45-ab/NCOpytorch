import os
import json
import pprint as pp
import numpy as np
import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger
from options import get_options
from train import train_epoch, get_inner_model
#from nets.attention_decor import AttentionModel
from nets.attention_decor import AttentionModel
from nets.encoder.gnn_encoder import GNNEncoder
from reinforcement_baseline import GeneticBaseline
from problem.problem_NCO import NCODataset, NCOProblem
from utils import torch_load_cpu


def run(opts):
    pp.pprint(vars(opts))
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    tb_logger = (
        None
        if opts.no_tensorboard
        else TbLogger(os.path.join(opts.log_dir, "nco", opts.run_name))
    )
    os.makedirs(opts.save_dir)
    with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
        json.dump(vars(opts), f, indent=True)
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")
    problem = NCOProblem()
    load_data = {}
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print(f"\nLoading data from {load_path}")
        load_data = torch_load_cpu(load_path)
    model = AttentionModel(
        problem=problem,
        embedding_dim=opts.embedding_dim,
        encoder_class=GNNEncoder,
        n_encode_layers=opts.n_encode_layers,
        n_heads=opts.n_heads,
        tanh_clipping=opts.tanh_clipping,
    ).to(opts.device)
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print(model)
    nb_param = sum(np.prod(list(param.data.size())) for param in model.parameters())
    print("Number of parameters: ", nb_param)
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get("model", {})})
    baseline = GeneticBaseline(problem, opts)
    if "baseline" in load_data:
        baseline.load_state_dict(load_data["baseline"])
    optimizer = optim.Adam([{"params": model.parameters(), "lr": opts.lr_model}])
    if "optimizer" in load_data:
        optimizer.load_state_dict(load_data["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: opts.lr_decay**epoch
    )
    val_datasets = [NCODataset(json_dir=opts.json_dir, num_samples=opts.val_size)]
    if opts.resume:
        epoch_resume = int(
            os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1]
        )
        torch.set_rng_state(load_data["rng_state"])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data["cuda_rng_state"])
        baseline.epoch_callback(model, epoch_resume)
        print(f"Resuming after {epoch_resume}")
        opts.epoch_start = epoch_resume + 1
    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        train_epoch(
            model,
            optimizer,
            baseline,
            lr_scheduler,
            epoch,
            val_datasets,
            problem,
            tb_logger,
            opts,
        )


if __name__ == "__main__":
    opts = get_options()
    opts.problem = "nco"
    opts.json_dir = "data/processed"
    opts.population_size = 50
    opts.generations = 100
    opts.mutation_rate = 0.1
    opts.crossover_rate = 0.8
    run(opts)
