# Benchmark Comparison Table — exp002_random_subset_pathmnist

> Evaluated across seeds: 42, 123, 456
> Format: **Mean ± Std** ($N=3$). Arrows indicate optimization direction: (↑) Higher is better, (↓) Lower is better.

| Unlearning Method      | Retain Accuracy (↑)   | Test Accuracy (↑)   | Forget Accuracy (↓)   | MIA Vulnerability (↓)   | Efficiency Ratio (↓)   |
|:-----------------------|:----------------------|:--------------------|:----------------------|:------------------------|:-----------------------|
| Exact Retrain (Oracle) | 0.9835 ± 0.0196       | 0.8739 ± 0.0141     | 0.9628 ± 0.0208       | 0.5884 ± 0.0400         | 1.0000 ± 0.0000        |
| Naive Fine-Tune        | 0.9995 ± 0.0003       | 0.8876 ± 0.0211     | 0.9989 ± 0.0006       | 0.6275 ± 0.0280         | 0.1670 ± 0.0007        |
| Gradient Ascent        | 0.8983 ± 0.0150       | 0.8200 ± 0.0283     | 0.8860 ± 0.0123       | 0.6062 ± 0.0158         | 0.0650 ± 0.0021        |

### Interpretation Notes:
- **Exact Retrain (Oracle)**: Gold standard reference; completely isolates and trains exclusively on retain data.
- **Naive Fine-Tune**: Fast but weak baseline; demonstrates catastrophic retention of forget-set memory.
- **Gradient Ascent**: Selective parameter-efficient unlearning; balances computational speedup against retention stability.
