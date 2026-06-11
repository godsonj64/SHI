# Initial smoke result

Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmark.py --epochs 1 --seeds 0 --limit-train 64 --limit-test 64 --methods erm shi --batch-size 32 --latent-dim 16
```

Result:

| method | IID accuracy | OOD accuracy | active dims |
|---|---:|---:|---:|
| ERM | 0.8125 | 0.140625 | 10.5 |
| SHI | 0.8125 | 0.140625 | 11.0 |

Interpretation: this is a smoke test only. It verifies that the real-image shifted benchmark and all training code run. It is not a final benchmark. Use the default command with more epochs and seeds for a meaningful comparison.
