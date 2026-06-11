from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from shi.data import make_loaders
from shi.train import train_method


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run SHI on Colored MNIST / real digit domain shift.')
    p.add_argument('--data-root', type=str, default='data')
    p.add_argument('--out-dir', type=str, default='results')
    p.add_argument('--methods', nargs='+', default=['erm', 'ib', 'irm', 'shi'], choices=['erm', 'ib', 'irm', 'shi'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--latent-dim', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--limit-train', type=int, default=2000)
    p.add_argument('--limit-test', type=int, default=1000)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--lambda-env', type=float, default=1.0)
    p.add_argument('--lambda-mut', type=float, default=1.0)
    p.add_argument('--lambda-comp', type=float, default=1e-3)
    p.add_argument('--lambda-decorr', type=float, default=1e-2)
    p.add_argument('--lambda-var', type=float, default=1e-2)
    p.add_argument('--mutation-sigma', type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for seed in args.seeds:
        train_loaders, eval_loaders = make_loaders(args.data_root, args.batch_size, seed, args.limit_train, args.limit_test)
        for method in args.methods:
            metrics = train_method(method, train_loaders, eval_loaders, seed, args.epochs, args.lr, args.weight_decay, args.latent_dim, args.device, args.lambda_env, args.lambda_mut, args.lambda_comp, args.lambda_decorr, args.lambda_var, args.mutation_sigma)
            rows.append(metrics)
            print(metrics)
    raw = pd.DataFrame(rows)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    raw.to_csv(Path(args.out_dir) / 'benchmark_raw.csv', index=False)
    metric_cols = [c for c in raw.columns if c.endswith('_acc')]
    summary = raw.groupby('method')[metric_cols].agg(['mean', 'std'])
    summary.to_csv(Path(args.out_dir) / 'benchmark_summary.csv')
    print(summary)
    if 'ood_corr_010_acc' in raw.columns:
        plt.figure(figsize=(7, 4))
        raw.groupby('method')['ood_corr_010_acc'].mean().sort_values(ascending=False).plot(kind='bar')
        plt.ylabel('OOD accuracy')
        plt.title('Colored MNIST OOD Accuracy')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(Path(args.out_dir) / 'ood_accuracy.png', dpi=180)


if __name__ == '__main__':
    main()
