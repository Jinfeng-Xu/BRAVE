"""BRAVE reflection allocator.

This module implements the paper's training-time allocation rule:

    g_BRAVE(u) = g_frag(u) + eta * g_conf(u) * 1[g_frag(u) >= q_tau].

It intentionally avoids LLM calls. The allocator only needs cached/full tool
scores and the user window used by the agent stage.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from tool_config import expand_tool_pool
from utils import load_dict, load_embeddings, mask_train_items


def normalize_scores(scores_by_tool):
    """Min-max normalize each user-tool score vector to [0, 1]."""
    scores = scores_by_tool.detach().clone().float()
    valid = torch.isfinite(scores)
    min_val = torch.where(valid, scores, torch.full_like(scores, float("inf"))).min(dim=2, keepdim=True).values
    max_val = torch.where(valid, scores, torch.full_like(scores, float("-inf"))).max(dim=2, keepdim=True).values
    scale = max_val - min_val
    scale[scale == 0] = 1.0
    normalized = (scores - min_val) / scale
    scores[valid] = normalized[valid]
    scores[~valid] = float("-inf")
    return scores


def ranking_fragility(normalized_scores, top_k=20):
    """Compute negative average top-1/top-2 margins across tools."""
    del top_k
    top2 = torch.topk(normalized_scores, k=2, dim=2).values
    margins = top2[:, :, 0] - top2[:, :, 1]
    return (-margins.mean(dim=1)).cpu().numpy()


def tool_conflict(normalized_scores, top_k=20):
    """Compute one minus average pairwise top-K Jaccard overlap."""
    top_items = torch.topk(normalized_scores, k=top_k, dim=2).indices.cpu().numpy()
    num_users, num_tools, _ = top_items.shape
    conflict = np.zeros(num_users, dtype=np.float32)
    if num_tools < 2:
        return conflict
    for user_idx in range(num_users):
        jaccards = []
        for a in range(num_tools):
            set_a = set(top_items[user_idx, a].tolist())
            for b in range(a + 1, num_tools):
                set_b = set(top_items[user_idx, b].tolist())
                union = max(1, len(set_a | set_b))
                jaccards.append(len(set_a & set_b) / union)
        conflict[user_idx] = 1.0 - float(np.mean(jaccards))
    return conflict


def aggregate_scores(normalized_scores, weights):
    return (normalized_scores * weights[:, :, None]).sum(dim=1)


def aggregate_margin_scores(normalized_scores, weights, boundary_k=20):
    """Aggregate-score boundary baselines used in the ablation study."""
    aggregated = aggregate_scores(normalized_scores, weights)
    top2 = torch.topk(aggregated, k=2, dim=1).values
    top_margin = -(top2[:, 0] - top2[:, 1]).cpu().numpy()
    topk = torch.topk(aggregated, k=boundary_k + 1, dim=1).values
    boundary = -(topk[:, boundary_k - 1] - topk[:, boundary_k]).cpu().numpy()
    return top_margin, boundary


def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    std = x.std()
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def initial_weights(num_users, num_tools):
    return torch.full((num_users, num_tools), 1.0 / max(1, num_tools), dtype=torch.float32)


def brave_scores(
    scores_by_tool,
    method="brave",
    budget=0.6,
    eta=0.25,
    tau=0.5,
    top_k=20,
    train_dict=None,
    user_ids=None,
    seed=42,
):
    """Return allocator scores and per-user diagnostics."""
    del budget
    normalized = normalize_scores(scores_by_tool)
    num_users, num_tools, _ = normalized.shape
    fragility = ranking_fragility(normalized, top_k=top_k)
    conflict = tool_conflict(normalized, top_k=top_k)
    gate_threshold = float(np.quantile(fragility, tau))
    gated = fragility + eta * conflict * (fragility >= gate_threshold)

    weights = initial_weights(num_users, num_tools).to(normalized.device)
    agg_margin, topk_boundary = aggregate_margin_scores(normalized, weights, boundary_k=top_k)
    history_len = np.zeros(num_users, dtype=np.float32)
    if train_dict is not None and user_ids is not None:
        history_len = np.asarray([len(train_dict.get(int(u), [])) for u in user_ids], dtype=np.float32)

    method = method.lower()
    if method == "brave":
        score = gated
    elif method in {"fragility", "fragility_only", "margin"}:
        score = fragility
    elif method in {"conflict", "conflict_only", "disagreement"}:
        score = conflict
    elif method in {"ungated", "ungated_additive"}:
        score = zscore(fragility) + eta * zscore(conflict)
    elif method in {"agg_margin", "aggregate_margin"}:
        score = agg_margin
    elif method in {"topk_boundary", "top_k_boundary"}:
        score = topk_boundary
    elif method in {"history", "histlen"}:
        score = -history_len
    elif method == "random":
        rng = np.random.default_rng(seed)
        score = rng.random(num_users)
    else:
        raise ValueError(f"Unknown allocator method: {method}")

    diagnostics = []
    for idx in range(num_users):
        diagnostics.append({
            "user_id": int(user_ids[idx]) if user_ids is not None else idx,
            "score": float(score[idx]),
            "g_frag": float(fragility[idx]),
            "g_conf": float(conflict[idx]),
            "g_brave": float(gated[idx]),
            "gate": int(fragility[idx] >= gate_threshold),
            "agg_margin_score": float(agg_margin[idx]),
            "topk_boundary_score": float(topk_boundary[idx]),
            "history_len": int(history_len[idx]),
        })
    return score, diagnostics, gate_threshold


def select_by_score(user_ids, score, budget):
    n = len(user_ids)
    k = int(math.ceil(float(budget) * n))
    k = max(0, min(k, n))
    if k == 0:
        return []
    order = np.argsort(-np.asarray(score))[:k]
    return [int(user_ids[int(i)]) for i in order]


def allocate_from_ranking_scores(
    ranking_score_dict,
    user_ids,
    budget=0.6,
    eta=0.25,
    tau=0.5,
    top_k=20,
    method="brave",
    train_dict=None,
    seed=42,
):
    tool_scores = [ranking_score_dict[tool][user_ids] for tool in ranking_score_dict]
    scores_by_tool = torch.stack(tool_scores, dim=1)
    score, diagnostics, gate_threshold = brave_scores(
        scores_by_tool,
        method=method,
        budget=budget,
        eta=eta,
        tau=tau,
        top_k=top_k,
        train_dict=train_dict,
        user_ids=user_ids,
        seed=seed,
    )
    selected = select_by_score(user_ids, score, budget)
    return selected, diagnostics, gate_threshold


def save_allocation(selected_user_ids, diagnostics, out_file):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for user_id in selected_user_ids:
            f.write(f"{int(user_id)}\n")
    diag_file = out_file.with_suffix(".diagnostics.csv")
    with open(diag_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(diagnostics[0].keys()) if diagnostics else ["user_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics)
    return out_file, diag_file


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BRAVE-selected user windows without LLM calls.")
    parser.add_argument("--dataset", default="amazon")
    parser.add_argument("--rec_tool", nargs="+", default=["ultra"], help="Tool names or pool aliases: simple, ultra")
    parser.add_argument("--embedding_root", default=None)
    parser.add_argument("--budget", type=float, default=0.6)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--method", default="brave",
                        choices=["brave", "fragility", "conflict", "ungated", "agg_margin", "topk_boundary", "history", "random"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batch", type=int, default=40)
    parser.add_argument("--start_batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_file", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    tools = expand_tool_pool(args.rec_tool)
    data_root = Path("data") / args.dataset
    if not (data_root / "train.txt").exists():
        data_root = Path("LightGCN-PyTorch/data") / args.dataset
    train_dict = load_dict(data_root / "train.txt")
    ranking_score_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for tool in tools:
        user_emb, item_emb = load_embeddings(tool, args.dataset, device, embedding_root=args.embedding_root)
        scores = mask_train_items(user_emb @ item_emb.t(), train_dict)
        ranking_score_dict[tool] = scores

    all_user_ids = sorted(train_dict.keys())
    start = (args.start_batch - 1) * args.batch_size
    end = start + args.batch_size * args.num_batch
    user_ids = all_user_ids[start:end]

    selected, diagnostics, gate_threshold = allocate_from_ranking_scores(
        ranking_score_dict,
        user_ids=user_ids,
        budget=args.budget,
        eta=args.eta,
        tau=args.tau,
        top_k=args.top_k,
        method=args.method,
        train_dict=train_dict,
        seed=args.seed,
    )
    out_file = args.out_file or (
        f"results/brave_selected_users/{args.dataset}_{'-'.join(tools)}_"
        f"{args.method}_b{args.budget:.2f}_eta{args.eta:g}_tau{args.tau:g}.txt"
    )
    selected_path, diag_path = save_allocation(selected, diagnostics, out_file)
    print(f"Selected {len(selected)} / {len(user_ids)} users; gate_threshold={gate_threshold:.6f}")
    print(f"Wrote {selected_path}")
    print(f"Wrote {diag_path}")


if __name__ == "__main__":
    main()
