import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, roc_auc_score

from data_load import data_load
from models import SJNPP
from predict import predict
from train import train_model
from utils import create_dir, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate SJ-NPP.")

    parser.add_argument("--data_file", type=str, default="./data/transformed_data_processed.csv")
    parser.add_argument("--item_embedding_file", type=str, default="./data/item_embeddings.pt")
    parser.add_argument("--unit", type=str, default="day", choices=["no", "day", "hour", "min"])
    parser.add_argument("--train_split", type=float, default=1.0)
    parser.add_argument("--valid_split", type=float, default=0.0)
    parser.add_argument("--dev", action="store_true", help="Use a small subset of sequences for a quick smoke run.")

    parser.add_argument("--encoder", type=str, default="gru", choices=["gru", "lstm"])
    parser.add_argument("--emsize", type=int, default=256)
    parser.add_argument("--nhid", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=1)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_topic", type=int, default=70)
    parser.add_argument("--item_emsize", type=int, default=128)
    parser.add_argument("--num_type", type=int, default=4)
    parser.add_argument("--num_prod", type=int, default=None)
    parser.add_argument("--action_encoding", type=str, default="index", choices=["index", "one-hot"])
    parser.add_argument("--self_embedding", action="store_true")

    parser.add_argument("--loss", type=str, default="all", choices=["time", "time+action", "time+item", "all"])
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--penalty", type=float, default=1e-4)
    parser.add_argument("--criterion", type=str, default="train")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--result_dir", type=str, default="./result/")
    parser.add_argument("--ckpt_dir", type=str, default="./models/")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the saved model after training.")
    parser.add_argument("--eval_up_lims", type=float, nargs="+", default=[5, 10, 20, 30, 40, 50])

    return parser.parse_args()


def build_run_name(args):
    if args.run_name:
        return args.run_name
    parts = [
        "sj_npp",
        str(args.num_topic),
        str(args.epochs),
        str(args.lr),
        str(args.penalty),
        str(args.emsize),
        str(args.nhid),
        str(args.dropout),
    ]
    return "_".join(parts)


def load_data(args, device):
    data = data_load(vars(args), dev=args.dev)
    data["num_seq"] = int(data["num_seq"])
    data["num_type"] = int(data["num_type"])

    for key, value in data.items():
        if torch.is_tensor(value) and key not in {"train_split_idx", "valid_split_idx", "eff_seqlen"}:
            data[key] = value.float().to(device)

    return data


def load_item_embeddings(args, device):
    if args.self_embedding:
        if args.num_prod is None:
            raise ValueError("--num_prod is required when --self_embedding is enabled.")
        return None

    vocab_emb = torch.load(args.item_embedding_file, map_location=device)
    if vocab_emb.shape[1] != args.item_emsize:
        raise ValueError(
            f"Item embedding dimension mismatch: expected {args.item_emsize}, got {vocab_emb.shape[1]}."
        )
    return vocab_emb.to(device)


def build_model(args, data, vocab_emb, device):
    num_prod = args.num_prod if args.num_prod is not None else vocab_emb.shape[0]
    return SJNPP({
        "num_seq": data["num_seq"],
        "num_type": args.num_type or data["num_type"],
        "nhid": args.nhid,
        "nhead": args.nhead,
        "nlayers": args.nlayers,
        "emsize": args.emsize,
        "dropout": args.dropout,
        "encoder": args.encoder,
        "device": device,
        "num_topic": args.num_topic,
        "item_emsize": args.item_emsize,
        "action_encoding": args.action_encoding,
        "self_embedding": args.self_embedding,
        "num_prod": num_prod,
        "vocab_emb": vocab_emb,
    }).to(device)


def evaluate_model(model, data, args):
    model.eval()
    for up_lim in args.eval_up_lims:
        ndcg_scores = []
        hit_scores = []
        pred_time_list = []
        pred_action_list = []
        pred_action_prob_list = []
        true_time_list = []
        true_action_list = []

        for user_idx in range(data["num_seq"]):
            target_idx = int(data["train_split_idx"][user_idx].detach().cpu().item())
            if target_idx >= int(data["eff_seqlen"][user_idx].detach().cpu().item()):
                continue

            true_time = data["time"][user_idx, target_idx]
            true_action = data["action"][user_idx, target_idx]
            true_item = data["item"][user_idx, target_idx]

            next_time, next_action, next_action_prob, _, _, next_item_prob = predict(model, data, user_idx, up_lim)
            pred_time_list.append(next_time)
            pred_action_list.append(int(next_action.detach().cpu().item()))
            pred_action_prob_list.append(next_action_prob.detach().cpu().tolist())
            true_time_list.append(float(true_time.detach().cpu().item()))
            true_action_list.append(int(true_action.detach().cpu().item()))

            recommended_items = torch.topk(next_item_prob, k=10)[1].detach().cpu().tolist()
            true_item_id = int(true_item.detach().cpu().item())
            hit_scores.append(1 if true_item_id in recommended_items else 0)
            ndcg_scores.append(0 if true_item_id not in recommended_items else 1 / np.log2(recommended_items.index(true_item_id) + 2))

        if not true_time_list:
            print(f"No valid evaluation events for up_lim={up_lim}.")
            continue

        print(f"Evaluation horizon: {up_lim}")
        print(f"SJ-NPP - rmse: {np.sqrt(mean_squared_error(true_time_list, pred_time_list)):.4f}")
        print(f"SJ-NPP - mae: {mean_absolute_error(true_time_list, pred_time_list):.4f}")
        print(f"SJ-NPP - accuracy: {accuracy_score(true_action_list, pred_action_list):.4f}")
        print(f"SJ-NPP - precision: {precision_score(true_action_list, pred_action_list, average='weighted', zero_division=0):.4f}")
        print(f"SJ-NPP - recall: {recall_score(true_action_list, pred_action_list, average='weighted', zero_division=0):.4f}")
        print(f"SJ-NPP - f1: {f1_score(true_action_list, pred_action_list, average='weighted', zero_division=0):.4f}")
        try:
            roc_auc = roc_auc_score(true_action_list, pred_action_prob_list, average="weighted", multi_class="ovr")
            print(f"SJ-NPP - roc_auc: {roc_auc:.4f}")
        except ValueError as exc:
            print(f"SJ-NPP - roc_auc: skipped ({exc})")
        print(f"SJ-NPP - NDCG@10: {np.mean(ndcg_scores):.4f}, HR@10: {np.mean(hit_scores):.4f}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    args.run_name = build_run_name(args)
    args.result_fn = f"{args.run_name}.pkl"
    args.ckpt_fn = f"{args.run_name}.ckpt"

    create_dir(args.result_dir)
    create_dir(args.ckpt_dir)
    setup_seed(args.seed)

    data = load_data(args, device)
    vocab_emb = load_item_embeddings(args, device)
    model = build_model(args, data, vocab_emb, device)
    model, _ = train_model(data, model, args)

    if args.evaluate:
        checkpoint = Path(args.ckpt_dir) / args.ckpt_fn
        model = torch.load(checkpoint, map_location=device)
        evaluate_model(model, data, args)


if __name__ == "__main__":
    main()
