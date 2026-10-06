"""
medforget/runner.py
--------------------
Main execution engine for MedForget-bench.

Reads an experiment YAML config, loops over all seeds and methods,
runs the full unlearning pipeline, evaluates all metrics, and saves
one JSON file per method per seed to the configured output directory.

Expected output structure (from YAML output_dir):
    results/exp001_classwise_pathmnist/exact_retrain_seed42.json
    results/exp001_classwise_pathmnist/naive_finetune_seed42.json
    ...
"""

import json
import time
from pathlib import Path

import torch
import yaml

from medforget.datasets.pathmnist import PathMNISTDataset
from medforget.scenarios.classwise import ClasswiseForgetScenario
from medforget.scenarios.random_subset import RandomSubsetForgetScenario
from medforget.models.resnet18 import build_resnet18
from medforget.methods.exact_retrain import ExactRetrain
from medforget.methods.finetune import NaiveFineTune
from medforget.metrics.utility import RetainAccuracy, TestAccuracy
from medforget.metrics.forgetting import ForgetAccuracy
from medforget.metrics.mia import LossThresholdMIA
from medforget.metrics.efficiency import EfficiencyRatio
from medforget.utils.seed import set_seed


# ── Plugin registries ─────────────────────────────────────────────────────────
# Maps config-file name strings to Python classes.
# To add a new dataset or scenario, just add an entry here — runner stays unchanged.

_DATASET_REGISTRY = {
    "pathmnist": PathMNISTDataset,
}

_SCENARIO_REGISTRY = {
    "classwise": ClasswiseForgetScenario,
    "random_subset": RandomSubsetForgetScenario,
}

_METHOD_REGISTRY = {
    "exact_retrain": ExactRetrain,
    "naive_finetune": NaiveFineTune,
}

_METRIC_REGISTRY = {
    "retain_acc": RetainAccuracy,
    "test_acc": TestAccuracy,
    "forget_acc": ForgetAccuracy,
    "mia": LossThresholdMIA,
    # "efficiency" handled separately — different compute() signature
}


def _build_method(method_cfg: dict, model_factory, batch_size: int, num_workers: int):
    """Instantiate an unlearning method from its YAML config block.

    Returns None for methods not yet implemented (will be skipped by runner).
    """
    name = method_cfg["name"]
    epochs = method_cfg.get("epochs", 30)  # exact_retrain has no epochs in YAML; default 30
    lr = float(method_cfg.get("lr", 1e-3))

    if name not in _METHOD_REGISTRY:
        return None  # Phase 6 methods (gradient_ascent, etc.) are not yet implemented

    if name == "exact_retrain":
        return _METHOD_REGISTRY[name](
            model_factory=model_factory,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif name == "naive_finetune":
        return _METHOD_REGISTRY[name](
            num_epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            num_workers=num_workers,
        )


def run_experiment(config_path: str) -> None:
    """Execute a full unlearning experiment from a YAML config file.

    Args:
        config_path: Path to the experiment YAML config.
    """
    # ── 1. Load Config ────────────────────────────────────────────────────────
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    exp_name = config["experiment"]["name"]
    seeds = config["experiment"]["seeds"]       # list: [42, 123, 456]
    output_dir = Path(config["output_dir"])     # results/exp001_classwise_pathmnist
    output_dir.mkdir(parents=True, exist_ok=True)

    # batch_size and num_workers are not in the YAML; use sensible defaults
    BATCH_SIZE = 256
    NUM_WORKERS = 0   # keep 0 for Windows compatibility

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Experiment : {exp_name}")
    print(f"Seeds      : {seeds}")
    print(f"Output dir : {output_dir}")
    print(f"Device     : {device}")

    # ── 2. Dataset (built once — same data across all seeds) ─────────────────
    dataset_name = config["dataset"]["name"]
    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(_DATASET_REGISTRY)}")

    ds = _DATASET_REGISTRY[dataset_name]()
    test_ds = ds.get_test_dataset()

    # ── 3. Scenario ───────────────────────────────────────────────────────────
    scenario_cfg = config["scenario"]
    scenario_name = scenario_cfg["name"]
    if scenario_name not in _SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario '{scenario_name}'.")

    # Pass extra YAML keys (e.g. forget_class) to the scenario constructor
    scenario_kwargs = {k: v for k, v in scenario_cfg.items() if k != "name"}
    scenario = _SCENARIO_REGISTRY[scenario_name](**scenario_kwargs)

    # ── 4. Baseline checkpoint ────────────────────────────────────────────────
    baseline_ckpt = Path("checkpoints/baseline.pt")
    if not baseline_ckpt.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found at '{baseline_ckpt}'.\n"
            "Run:  python scripts/train_baseline.py  first."
        )

    def get_baseline_model() -> torch.nn.Module:
        """Return a fresh baseline model loaded from the checkpoint."""
        m = build_resnet18(ds.num_classes)
        m.load_state_dict(torch.load(baseline_ckpt, map_location="cpu", weights_only=True))
        return m.to(device)

    model_factory = lambda: build_resnet18(ds.num_classes)

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    metric_names = [m for m in config["metrics"] if m != "efficiency"]
    
    # Check for invalid metrics in config
    for m in metric_names:
        if m not in _METRIC_REGISTRY:
            raise ValueError(f"Unknown metric '{m}' in config.")
            
    metrics = [
        _METRIC_REGISTRY[name](batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        for name in metric_names
    ]
    
    run_efficiency = "efficiency" in config["metrics"]
    if run_efficiency:
        efficiency_metric = EfficiencyRatio()

    # ── 6. Main Loop: Seeds × Methods ────────────────────────────────────────
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        set_seed(seed)

        # Re-apply scenario each seed so random-subset scenarios differ per seed.
        # For class-wise this has no effect, but keeps the loop consistent.
        train_ds = ds.get_train_dataset()
        forget_set, retain_set = scenario.apply(train_ds)
        print(f"Split: {len(forget_set)} forget | {len(retain_set)} retain | {len(test_ds)} test")

        retrain_time = 1.0   # overwritten when exact_retrain runs

        for method_cfg in config["methods"]:
            method_name = method_cfg["name"]

            method = _build_method(method_cfg, model_factory, BATCH_SIZE, NUM_WORKERS)
            if method is None:
                print(f"\n  [skip] '{method_name}' not yet implemented (Phase 6)")
                continue

            print(f"\nRunning: {method_name}")

            # Load a clean, untouched baseline for every method
            model = get_baseline_model()

            # Time the unlearning step
            t0 = time.time()
            unlearned_model = method.run(model, forget_set, retain_set)
            elapsed = time.time() - t0

            # ExactRetrain sets the efficiency reference for this seed
            if method_name == "exact_retrain":
                retrain_time = elapsed

            print(f"  Time: {elapsed:.2f}s")

            # Evaluate all metrics
            result = {
                "experiment": exp_name,
                "method": method_name,
                "seed": seed,
                "time_seconds": round(elapsed, 4),
            }

            for metric in metrics:
                score = metric.compute(unlearned_model, forget_set, retain_set, test_ds)
                result[metric.name] = round(score, 6)
                print(f"  {metric.name}: {score:.4f}")

            if run_efficiency:
                eff = efficiency_metric.compute(method_time=elapsed, retrain_time=retrain_time)
                result[efficiency_metric.name] = round(eff, 6)
                print(f"  {efficiency_metric.name}: {eff:.4f}")

            # Save: results/exp001_classwise_pathmnist/exact_retrain_seed42.json
            out_file = output_dir / f"{method_name}_seed{seed}.json"
            with open(out_file, "w") as f:
                json.dump(result, f, indent=4)
            print(f"  Saved → {out_file}")

    print(f"\nAll done. Results in: {output_dir}/")
