# SJ-NPP

This repository provides the model code for **SJ-NPP**, proposed in the paper **"When, How, and What: An Attention-Based Neural Point Process for Joint Modeling of E-commerce Event Sequences"**.

SJ-NPP is an attention-based neural point process for jointly modeling e-commerce event sequences. Given a user's historical events, the model predicts when the next event occurs, how the user acts, and what item is involved.

## Contents

- `models.py`: PyTorch implementation of the `SJNPP` model.
- `main.py`: Command-line entry point for training and optional evaluation.
- `loss.py`: Time, action, and item negative log-likelihood losses.
- `train.py`: Training loop and early stopping integration.
- `predict.py`: Next-event time, action, and item prediction utilities.
- `data_load.py`: CSV-to-tensor data loading and sequence split helpers.
- `utils.py`: Argument parsing, reproducibility, timing, and checkpoint helpers.

The repository focuses on the reusable model implementation. Large datasets, trained checkpoints, notebook outputs, and generated result files are not included.

## Expected Data Format

`data_load.py` expects a CSV file with at least the following columns:

- `User_id`: zero-based user/sequence id.
- `Event_id`: event order within a user sequence.
- `Time`: event timestamp or elapsed time.
- `Action`: event/action type id.
- `Item_id`: item id used by the event.

The loader pads shorter sequences with `9999999` and returns tensors for `time`, `action`, `item`, `delta_t`, `elapsed_t`, plus train/validation split indices.

## Basic Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train SJ-NPP from the command line:

```bash
python main.py --data_file ./data/your_processed_sequences.csv --item_embedding_file ./data/item_embeddings.pt
```

Train and evaluate next-event predictions after saving the best checkpoint:

```bash
python main.py --data_file ./data/your_processed_sequences.csv --item_embedding_file ./data/item_embeddings.pt --evaluate
```

Example skeleton:

```python
import torch
from data_load import data_load
from models import SJNPP
from train import train_model

args = {
    "data_file": "./data/your_processed_sequences.csv",
    "train_split": 0.6,
    "valid_split": 0.2,
    "unit": "day",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "encoder": "gru",
    "emsize": 128,
    "nhid": 128,
    "nlayers": 1,
    "dropout": 0.4,
    "loss": "all",
    "lr": 1e-4,
    "penalty": 1e-4,
    "epochs": 100,
    "patience": 10,
    "criterion": "valid",
    "ckpt_dir": "./models/",
    "ckpt_fn": "sj_npp.ckpt",
    "result_dir": "./result/",
    "result_fn": "training_log.pkl",
}

data = data_load(args, dev=False)

# Replace this with pre-trained item embeddings or set self_embedding=True.
vocab_emb = torch.randn(1000, 128, device=args["device"])

model = SJNPP({
    "num_seq": data["num_seq"],
    "num_type": int(data["num_type"]),
    "num_prod": vocab_emb.shape[0],
    "num_topic": 70,
    "item_emsize": vocab_emb.shape[1],
    "vocab_emb": vocab_emb,
    "self_embedding": False,
    "action_encoding": "index",
    "nhead": 1,
    "encoder": args["encoder"],
    "emsize": args["emsize"],
    "nhid": args["nhid"],
    "nlayers": args["nlayers"],
    "dropout": args["dropout"],
    "device": args["device"],
}).to(args["device"])

model, logs = train_model(data, model, args)
```

## Citation

If you use this code in your research, please cite the paper:

```bibtex
@article{sj_npp,
    title = {When, How, and What: An Attention-Based Neural Point Process for Joint Modeling of E-commerce Event Sequences},
    author = {Author information to be added},
    journal = {Publication information to be added},
    year = {Year to be added}
}
```

## Repository Scope

This public repository contains the core SJ-NPP implementation and supporting utilities. Dataset-specific preprocessing, private data files, trained model checkpoints, and experiment outputs should be managed separately according to the data release policy of the corresponding study.
