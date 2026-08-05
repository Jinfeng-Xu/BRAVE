#!/usr/bin/env python3
"""Offline probe for selective reflection in PRTA-style tool learning.

This script asks a cheap question before spending LLM calls:
Can unsupervised self-doubt signals identify users who benefit most from
tool-weight reflection?
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tool_config import expand_tool_pool, embedding_path as registry_embedding_path


REC_TOOLS = ("lgn", "srec", "smpx")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="amazon")
    parser.add_argument("--rec_tool", nargs="+", default=["simple"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--random_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batch", type=int, default=40)
    parser.add_argument("--start_batch", type=int, default=145)
    parser.add_argument("--user_scope", choices=["agent_window", "all"], default="agent_window")
    parser.add_argument("--out_dir", default="results/selective_reflection")
    return parser.parse_args()


def load_dict(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            data[int(parts[0])] = [int(x) for x in parts[1:]]
    return data


def load_single_item_dict(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                data[int(parts[0])] = int(parts[1])
    return data


def embedding_path(tool, dataset, kind):
    return registry_embedding_path(tool, dataset, kind)


def load_embeddings(tool, dataset, train_dict):
    user_emb = torch.load(embedding_path(tool, dataset, "user"), map_location="cpu", weights_only=True)
    item_emb = torch.load(embedding_path(tool, dataset, "item"), map_location="cpu", weights_only=True)
    if len(train_dict) == user_emb.shape[0] - 1:
        user_emb = user_emb[1:]
        item_emb = item_emb[1:]
    return user_emb.detach().float(), item_emb.detach().float()


def mask_train_items(scores, train_dict):
    scores = scores.clone()
    for user_id, items in train_dict.items():
        if user_id < scores.shape[0]:
            scores[user_id, items] = float("-inf")
    return scores


def normalize_rows(scores, min_val=-10.0, max_val=10.0):
    out = scores.clone()
    valid = torch.isfinite(out)
    masked_min = torch.where(valid, out, torch.full_like(out, float("inf"))).min(dim=1, keepdim=True).values
    masked_max = torch.where(valid, out, torch.full_like(out, float("-inf"))).max(dim=1, keepdim=True).values
    scale = masked_max - masked_min
    scale[scale == 0] = 1.0
    normalized = (out - masked_min) / scale * (max_val - min_val) + min_val
    out[valid] = normalized[valid]
    return out


def initial_weights(num_users, dataset, num_tools):
    if dataset == "amazon" and num_tools == 3:
        base = torch.tensor([0.2, 0.4, 0.4], dtype=torch.float32)
    else:
        base = torch.full((num_tools,), 1.0 / num_tools, dtype=torch.float32)
    return base.repeat(num_users, 1)


def ranks_for_items(scores_by_tool, item_ids):
    user_idx = torch.arange(scores_by_tool.shape[0])
    ranks = []
    for tool_idx in range(scores_by_tool.shape[1]):
        scores = scores_by_tool[:, tool_idx, :]
        target_scores = scores[user_idx, item_ids]
        ranks.append((scores > target_scores[:, None]).sum(dim=1) + 1)
    return torch.stack(ranks, dim=1).float()


def metric_per_user(scores, item_ids, k):
    topk = torch.topk(scores, k=k, dim=1).indices
    hits = topk.eq(item_ids[:, None])
    recall = hits.any(dim=1).float()
    ndcg = torch.zeros_like(recall)
    hit_positions = hits.nonzero(as_tuple=False)
    if hit_positions.numel() > 0:
        ndcg[hit_positions[:, 0]] = 1.0 / torch.log2(hit_positions[:, 1].float() + 2.0)
    return recall.detach().numpy(), ndcg.detach().numpy()


def aggregate_scores(scores_by_tool, weights):
    return (scores_by_tool * weights[:, :, None]).sum(dim=1)


def reflected_weights_from_valid_ranks(base_weights, valid_ranks, alpha):
    inv = 1.0 / valid_ranks.clamp_min(1.0)
    target = inv / inv.sum(dim=1, keepdim=True)
    return alpha * target + (1.0 - alpha) * base_weights


def topk_sets(scores_by_tool, k):
    return torch.topk(scores_by_tool, k=k, dim=2).indices.detach().numpy()


def pairwise_jaccard_features(topk_indices):
    num_users, num_tools, _ = topk_indices.shape
    avg_jaccard = np.zeros(num_users, dtype=np.float32)
    min_jaccard = np.ones(num_users, dtype=np.float32)
    for u in range(num_users):
        vals = []
        for a in range(num_tools):
            set_a = set(topk_indices[u, a].tolist())
            for b in range(a + 1, num_tools):
                set_b = set(topk_indices[u, b].tolist())
                vals.append(len(set_a & set_b) / max(1, len(set_a | set_b)))
        avg_jaccard[u] = float(np.mean(vals))
        min_jaccard[u] = float(np.min(vals))
    return avg_jaccard, min_jaccard


def margin_uncertainty(scores_by_tool):
    top2 = torch.topk(scores_by_tool, k=2, dim=2).values
    margins = (top2[:, :, 0] - top2[:, :, 1]).detach().numpy()
    return -margins.mean(axis=1), -margins.min(axis=1)


def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    std = x.std()
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def select_indices(score, budget):
    n = len(score)
    m = int(round(n * budget))
    if m <= 0:
        return np.array([], dtype=np.int64)
    if m >= n:
        return np.arange(n, dtype=np.int64)
    return np.argsort(-score)[:m]


def evaluate_budget_curve(scores_by_tool, base_weights, reflected_weights, test_items, selectors, budgets, k, random_trials, seed):
    rows = []
    n = scores_by_tool.shape[0]
    rng = np.random.default_rng(seed)
    for selector_name, selector_score in selectors.items():
        for budget in budgets:
            if selector_name == "random":
                recall_vals, ndcg_vals = [], []
                m = int(round(n * budget))
                for _ in range(random_trials):
                    chosen = rng.choice(n, size=m, replace=False) if m > 0 else np.array([], dtype=np.int64)
                    weights = base_weights.clone()
                    if len(chosen):
                        weights[chosen] = reflected_weights[chosen]
                    scores = aggregate_scores(scores_by_tool, weights)
                    recall, ndcg = metric_per_user(scores, test_items, k)
                    recall_vals.append(recall.mean())
                    ndcg_vals.append(ndcg.mean())
                rows.append([selector_name, budget, float(np.mean(recall_vals)), float(np.mean(ndcg_vals))])
            else:
                chosen = select_indices(selector_score, budget)
                weights = base_weights.clone()
                if len(chosen):
                    weights[chosen] = reflected_weights[chosen]
                scores = aggregate_scores(scores_by_tool, weights)
                recall, ndcg = metric_per_user(scores, test_items, k)
                rows.append([selector_name, budget, float(recall.mean()), float(ndcg.mean())])
    return rows


def main():
    args = parse_args()
    rec_tools = expand_tool_pool(args.rec_tool)
    data_root = Path("data") / args.dataset
    if not (data_root / "train.txt").exists():
        data_root = Path("LightGCN-PyTorch/data") / args.dataset
    train_dict = load_dict(data_root / "train.txt")
    valid_dict = load_single_item_dict(Path("data") / args.dataset / "valid.txt")
    test_dict = load_single_item_dict(Path("data") / args.dataset / "test.txt")

    ranking_scores = []
    for tool in rec_tools:
        user_emb, item_emb = load_embeddings(tool, args.dataset, train_dict)
        scores = mask_train_items(user_emb @ item_emb.t(), train_dict)
        ranking_scores.append(normalize_rows(scores))
    scores_by_tool = torch.stack(ranking_scores, dim=1)

    total_users = scores_by_tool.shape[0]
    if args.user_scope == "agent_window":
        start_user = (args.start_batch - 1) * args.batch_size
        end_user = start_user + args.batch_size * args.num_batch
        user_ids = list(range(start_user, min(end_user, total_users)))
    else:
        user_ids = list(range(total_users))

    scores_by_tool = scores_by_tool[user_ids]
    num_users = scores_by_tool.shape[0]
    valid_items = torch.tensor([valid_dict[u] for u in user_ids], dtype=torch.long)
    test_items = torch.tensor([test_dict[u] for u in user_ids], dtype=torch.long)
    base_weights = initial_weights(num_users, args.dataset, len(rec_tools))

    valid_ranks = ranks_for_items(scores_by_tool, valid_items)
    reflected_weights = reflected_weights_from_valid_ranks(base_weights, valid_ranks, args.alpha)

    base_scores = aggregate_scores(scores_by_tool, base_weights)
    reflected_scores = aggregate_scores(scores_by_tool, reflected_weights)
    _, base_valid_ndcg = metric_per_user(base_scores, valid_items, args.top_k)
    _, reflected_valid_ndcg = metric_per_user(reflected_scores, valid_items, args.top_k)
    valid_gain = reflected_valid_ndcg - base_valid_ndcg

    topk_indices = topk_sets(scores_by_tool, args.top_k)
    avg_jaccard, min_jaccard = pairwise_jaccard_features(topk_indices)
    mean_margin_unc, min_margin_unc = margin_uncertainty(scores_by_tool)
    history_len = np.array([len(train_dict.get(u, [])) for u in user_ids], dtype=np.float32)
    disagreement = 1.0 - avg_jaccard
    hard_disagreement = 1.0 - min_jaccard
    hybrid = zscore(disagreement) + 0.7 * zscore(mean_margin_unc) + 0.3 * zscore(history_len)

    budgets = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0]
    selectors = {
        "random": np.zeros(num_users, dtype=np.float32),
        "disagreement": disagreement,
        "hard_disagreement": hard_disagreement,
        "score_uncertainty": mean_margin_unc,
        "hybrid_self_doubt": hybrid,
        "oracle_valid_gain": valid_gain,
    }
    rows = evaluate_budget_curve(
        scores_by_tool,
        base_weights,
        reflected_weights,
        test_items,
        selectors,
        budgets,
        args.top_k,
        args.random_trials,
        args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / f"{args.dataset}_budget_curves.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["selector", "budget", f"recall@{args.top_k}", f"ndcg@{args.top_k}"])
        writer.writerows(rows)

    feature_path = out_dir / f"{args.dataset}_user_features.csv"
    with open(feature_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "user_id",
            "history_len",
            "avg_topk_jaccard",
            "min_topk_jaccard",
            "disagreement",
            "mean_margin_uncertainty",
            "valid_proxy_gain",
        ])
        for u in range(num_users):
            writer.writerow([
                user_ids[u],
                int(history_len[u]),
                float(avg_jaccard[u]),
                float(min_jaccard[u]),
                float(disagreement[u]),
                float(mean_margin_unc[u]),
                float(valid_gain[u]),
            ])

    selected_dir = out_dir / f"{args.dataset}_selected_users"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for selector_name, selector_score in selectors.items():
        if selector_name == "random":
            continue
        for budget in budgets:
            chosen_positions = select_indices(selector_score, budget)
            selected_path = selected_dir / f"{selector_name}_budget_{budget:.2f}.txt"
            with open(selected_path, "w", encoding="utf-8") as f:
                for pos in chosen_positions:
                    f.write(f"{user_ids[int(pos)]}\n")

    summary_path = out_dir / f"{args.dataset}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"dataset={args.dataset}\n")
        f.write(f"user_scope={args.user_scope}\n")
        f.write(f"start_batch={args.start_batch}\n")
        f.write(f"num_batch={args.num_batch}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write("llm_model=gpt-5.6-luna\n")
        f.write("agent_window_note=matches the sampled-agent window used by the LLM-agent stage; no LLM calls are made in this offline proxy probe.\n")
        f.write(f"users={num_users}\n")
        for tool_idx, tool in enumerate(rec_tools):
            recall, ndcg = metric_per_user(scores_by_tool[:, tool_idx, :], test_items, args.top_k)
            f.write(f"{tool}: recall@{args.top_k}={recall.mean():.6f}, ndcg@{args.top_k}={ndcg.mean():.6f}\n")
        base_recall, base_ndcg = metric_per_user(base_scores, test_items, args.top_k)
        full_recall, full_ndcg = metric_per_user(reflected_scores, test_items, args.top_k)
        f.write(f"base_weighted: recall@{args.top_k}={base_recall.mean():.6f}, ndcg@{args.top_k}={base_ndcg.mean():.6f}\n")
        f.write(f"full_proxy_reflection: recall@{args.top_k}={full_recall.mean():.6f}, ndcg@{args.top_k}={full_ndcg.mean():.6f}\n")
        f.write(f"mean_valid_proxy_gain={valid_gain.mean():.6f}\n")
        f.write(f"curve_csv={curve_path}\n")
        f.write(f"feature_csv={feature_path}\n")
        f.write(f"selected_user_dir={selected_dir}\n")

    print(f"Wrote {curve_path}")
    print(f"Wrote {feature_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
