# Benchmark Comparison Table — exp001_classwise_pathmnist

> Evaluated across seeds: 42, 123, 456
> Format: **Mean ± Std** ($N=3$). Arrows indicate optimization direction: (↑) Higher is better, (↓) Lower is better.

| Unlearning Method      | Retain Accuracy (↑)   | Test Accuracy (↑)   | Forget Accuracy (↓)   | MIA Vulnerability (↓)   | Efficiency Ratio (↓)   |
|:-----------------------|:----------------------|:--------------------|:----------------------|:------------------------|:-----------------------|
| Exact Retrain (Oracle) | 0.9968 ± 0.0004       | 0.7995 ± 0.0255     | 0.0000 ± 0.0000       | 0.0469 ± 0.0032         | 1.0000 ± 0.0000        |
| Naive Fine-Tune        | 0.9961 ± 0.0023       | 0.8820 ± 0.0220     | 0.9308 ± 0.0336       | 0.4505 ± 0.0396         | 0.1668 ± 0.0002        |
| Gradient Ascent        | 0.8154 ± 0.0024       | 0.4785 ± 0.0099     | 0.0000 ± 0.0000       | 0.0786 ± 0.0034         | 0.0603 ± 0.0010        |

### Interpretation Notes:
- **Exact Retrain (Oracle)**: Gold standard reference; completely isolates and trains exclusively on retain data.
- **Naive Fine-Tune**: Fast but weak baseline; demonstrates catastrophic retention of forget-set memory.
- **Gradient Ascent**: Selective parameter-efficient unlearning; balances computational speedup against retention stability.
