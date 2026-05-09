# Experiments

These scripts reproduce the main table/figure in the report.

## Recommended: run on Colab

Open `notebooks/colab_train.ipynb` and run the notebook top-to-bottom. It:

- downloads NoduleMNIST3D into `data/`
- runs the 4 experiment rows with 3 seeds each
- aggregates into `results/main.csv`
- generates a saliency PDF panel

## Running from the command line (GPU recommended)

First download the dataset:

```bash
python data/download_medmnist.py --out data/raw
```

Then run:

```bash
bash experiments/run_all.sh
```

Each script writes per-seed outputs to `results/<run_name>/` and stores a `metrics.json`.
