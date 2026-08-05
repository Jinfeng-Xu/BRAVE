#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import itertools
import json
import random
import os
import sys
import numpy as np
from tqdm import tqdm
import concurrent.futures
from torch.utils.data import Dataset, DataLoader
from dataloader import RecommendationDataset, collate_fn, save_user_profile, load_user_profile
from utils import *
from memory_bank import UserMemoryBank
from prompt_bank import LLM_agent, PromptBank
from brave_allocator import allocate_from_ranking_scores, save_allocation
from tool_config import expand_tool_pool


# In[2]:


def parse_args():
    parser = argparse.ArgumentParser(description="Recommendation Model Configuration")

    # Define arguments
    parser.add_argument("--dataset", type=str, default="amazon", help="Dataset name")
    parser.add_argument("--rec_tool", nargs='+', default=['ultra'], help="Recommendation tools or pool aliases: simple, ultra")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--llm_model", type=str, default='gpt-5.6-luna', help="LLM controller model")
    parser.add_argument("--llm_model_profile", type=str, default=None, help="LLM model name for user profile generation")
    parser.add_argument("--llm_model_id", type=int, default=1, help="LLM model name")
    parser.add_argument("--local_eval", type=float, default=0.1, help='learning rate when using LLM to evalaute ranking tools independently')
    parser.add_argument("--local_eval_num", type=int, default=20, help='number of recalled items when using LLM to evalaute ranking tools independently')
    parser.add_argument("--llm_cmpr", type=float, default=0.1, help='learning rate when using LLM to compare ranking tools')
    parser.add_argument("--llm_cmpr_num", type=int, default=20, help='number of recalled items when using LLM to compare ranking tools')
    parser.add_argument("--rerank_infr", type=int, default=20, help='number of candidate items when using LLM to rerank aggeragated ranking list')
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_batch", type=int, default=40, help="Number of batches")
    parser.add_argument("--start_batch", type=int, default=145, help="Position of first training batch in dataset")
    parser.add_argument("--lr_decay", type=float, default=0.5, help="decay of learning rate each epoch")
    parser.add_argument("--patience", type=int, default=2,help="early stopping epochs")
    parser.add_argument("--eval_k", type=int, default=20)
    parser.add_argument("--save_weights", type=int, default=0)
    parser.add_argument("--selected_user_file", type=str, default=None, help="Optional file containing user ids allowed to use LLM reflection")
    parser.add_argument("--selective_rerank", type=int, default=0, help="If 1, also apply selected_user_file to LLM inference reranking")
    parser.add_argument("--profile_start_batch", type=int, default=1, help="First batch used when generating a missing user profile file")
    parser.add_argument("--profile_num_batch", type=int, default=None, help="Number of batches used when generating a missing user profile file")
    parser.add_argument("--embedding_root", type=str, default=None, help="Optional directory containing {tool}_{dataset}_{user,item}_embeddings.pt")
    parser.add_argument("--allocator", type=str, default="brave",
                        choices=["none", "selected_file", "brave", "fragility", "conflict", "ungated", "agg_margin", "topk_boundary", "history", "random"],
                        help="Training-time reflection allocator")
    parser.add_argument("--budget", type=float, default=0.6, help="Fraction of sampled-agent window receiving LLM reflection")
    parser.add_argument("--eta", type=float, default=0.25, help="BRAVE conflict weight")
    parser.add_argument("--tau", type=float, default=0.5, help="BRAVE fragility gate quantile")
    parser.add_argument("--allocator_top_k", type=int, default=20, help="Top-K used by allocator margins/disagreement")
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()
    
    return vars(args)


def load_selected_user_ids(path):
    if not path:
        return None
    selected = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                selected.add(int(line))
    return selected


def get_selected_positions(users, selected_user_ids):
    if selected_user_ids is None:
        return list(range(len(users)))
    return [idx for idx, user_id in enumerate(users) if int(user_id) in selected_user_ids]


def get_window_user_ids(dataset, start_batch, batch_size, num_batch):
    start = (start_batch - 1) * batch_size
    end = start + batch_size * num_batch
    return [int(user_id) for user_id, _ in dataset.data[start:end]]


def subset_list(values, positions):
    return [values[idx] for idx in positions]


def run_agent_with_batch_size(agent, prompts, batch_size, mode="re"):
    original_batch_size = agent.batch_size
    agent.batch_size = batch_size
    try:
        return agent.parallel_ranking_evaluation(prompts, mode=mode)
    finally:
        agent.batch_size = original_batch_size


# In[3]:


if __name__ == "__main__":
    args = parse_args()
    args['rec_tool'] = expand_tool_pool(args['rec_tool'])
    data_path = './data/%s/'%args['dataset']
    train_path = data_path + "train.txt"
    test_path = data_path + "test.txt"
    valid_path = data_path + "valid.txt"
    item_description_path = data_path + "item_descriptions.csv"
    dataset = args['dataset']
    batch_size = args['batch_size']
    top_k = args['top_k']
    local_eval_lr = args['local_eval']
    llm_cmpr_lr = args['llm_cmpr']
    selected_user_ids = load_selected_user_ids(args.get('selected_user_file'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dict, test_dict = load_train_test_sample(dataset)
    ranking_score_dict = {}
    for rec_tool in args['rec_tool']:
        user_emb, item_emb = load_embeddings(rec_tool, dataset, device, embedding_root=args.get('embedding_root'))
        user_emb, item_emb = remove_padding_user(user_emb, item_emb, train_dict)
        ground_truth = convert_test_dict_to_tensor(test_dict, user_emb.shape[0], item_emb.shape[0], device)
        ranking_scores = torch.matmul(user_emb, item_emb.t())
        ranking_scores = mask_train_items(ranking_scores, train_dict)
        ranking_score_dict[rec_tool] = ranking_scores

    recalled_item_num = max(args['local_eval_num'], args['llm_cmpr_num'])
    train_dataset = RecommendationDataset(train_path, valid_path, item_description_path, ranking_score_dict, recalled_item_num)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataset = RecommendationDataset(train_path, test_path, item_description_path, ranking_score_dict, top_k)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    user_num, item_num = len(train_dataset), item_emb.shape[0]
    user_memory_bank = UserMemoryBank(user_num, dataset, args['rec_tool'])
    prompt_bank = PromptBank(item_num, top_k)
    agent = LLM_agent(batch_size,args["llm_model"],args["llm_model_id"])
    batch_rec_tool = get_batch_rec_tool(train_dataset.rec_tool_order, batch_size)
    
    if ('llm_model_profile' not in args) or (args['llm_model_profile']) == None:
        args['llm_model_profile'] = args['llm_model']
    profile_path = data_path + f"user_profile_{args['llm_model_profile']}.csv"
    if not os.path.exists(profile_path):  
        save_user_profile(train_loader, profile_path, agent, user_memory_bank, batch_num=args['profile_num_batch'], start_batch=args['profile_start_batch'])
    else:
        print(f"{profile_path} already exists. No need to re-run.")
        user_memory_bank = load_user_profile(profile_path, user_memory_bank)

    
    print_log(args)
    start_batch = args['start_batch']
    k = args['num_batch']
    if args['allocator'] not in ["none", "selected_file"]:
        user_window = get_window_user_ids(train_dataset, start_batch, batch_size, k)
        selected, diagnostics, gate_threshold = allocate_from_ranking_scores(
            ranking_score_dict,
            user_ids=user_window,
            budget=args['budget'],
            eta=args['eta'],
            tau=args['tau'],
            top_k=args['allocator_top_k'],
            method=args['allocator'],
            train_dict=train_dict,
            seed=args['seed'],
        )
        selected_user_ids = set(selected)
        allocation_name = (
            f"log/{dataset}/selected_{'-'.join(args['rec_tool'])}_{args['allocator']}_"
            f"b{args['budget']:.2f}_eta{args['eta']:g}_tau{args['tau']:g}.txt"
        )
        selected_path, diag_path = save_allocation(selected, diagnostics, allocation_name)
        print(f"BRAVE allocation selected {len(selected_user_ids)} / {len(user_window)} users; gate_threshold={gate_threshold:.6f}")
        print(f"Saved selected users to {selected_path} and diagnostics to {diag_path}")
    elif args['allocator'] == "selected_file" and selected_user_ids is None:
        raise ValueError("--allocator selected_file requires --selected_user_file")
    if selected_user_ids is not None:
        print(f"Selective training-time LLM reflection enabled for {len(selected_user_ids)} users")
    best_ndcg, recall_at_best, patience_counter = 0, 0, 0
    for epoch in range(args['epochs']):
        ctr, recall, ndcg = 0, 0, 0
        for batch_idx, (users, items, valid_item, descriptions, valid_description, ranking_matrix, rec_top_item_descriptions) in tqdm(enumerate(itertools.islice(train_loader, start_batch - 1, start_batch - 1 + k)),total=k, desc="Training Batches"):
            normalized_ranking_matrix = normalize_list_of_tensors_row(ranking_matrix)
            # _, indices = torch.topk(normalized_ranking_matrix, top_k, dim=2)
            batch_ranking = get_ground_truth_ranking(normalized_ranking_matrix, torch.tensor(valid_item).to(device)) + 1
            batch_weight = user_memory_bank.get_weight_memory_batch(users).to(device)
            updated_weight = optimize_by_ranking(batch_ranking.to(torch.float32), batch_weight)
            batch_user_profiles = user_memory_bank.get_profile_memory_batch(users)
            selected_positions = get_selected_positions(users, selected_user_ids)
            if args['local_eval']:
                if selected_positions:
                    selected_profiles = subset_list(batch_user_profiles, selected_positions)
                    selected_valid_desc = subset_list(valid_description, selected_positions)
                    selected_rec_desc = subset_list(rec_top_item_descriptions, selected_positions)
                    if args['local_eval_num'] < recalled_item_num:
                        selected_rec_desc = slice_nested_list(selected_rec_desc,args['local_eval_num'])
                    rec_tool_evaluation_prompt = prompt_bank.get_item_relavant_evaluation_prompt(selected_profiles, selected_valid_desc, selected_rec_desc)
                    rec_tool_evaluation = run_agent_with_batch_size(agent, rec_tool_evaluation_prompt, len(selected_positions))
                    selected_weight = updated_weight[selected_positions]
                    updated_weight[selected_positions] = optimize_by_local_evaluation(rec_tool_evaluation, selected_weight, device=device, alpha=local_eval_lr)

            if args['llm_cmpr']:
                if selected_positions:
                    selected_valid_desc = subset_list(valid_description, selected_positions)
                    selected_rec_desc = subset_list(rec_top_item_descriptions, selected_positions)
                    if args['llm_cmpr_num'] < recalled_item_num:
                        selected_rec_desc = slice_nested_list(selected_rec_desc,args['llm_cmpr_num'])
                    item_comparison_prompt = prompt_bank.get_recall_item_comparison_prompt(selected_valid_desc, selected_rec_desc)
                    item_comparison_result = run_agent_with_batch_size(agent, item_comparison_prompt, len(selected_positions))
                    selected_weight = updated_weight[selected_positions]
                    updated_weight[selected_positions] = optimize_by_comparison(item_comparison_result, selected_weight, alpha=llm_cmpr_lr)
            user_memory_bank.batch_update(users, weights_memories=updated_weight)
            aggregate_ranking_score = weighted_sum(updated_weight.to(device),normalized_ranking_matrix)

            

            recall, ndcg, ctr = update_metrics(aggregate_ranking_score, valid_item, recall, ndcg, ctr, k=top_k, device=device)
        print('Train Epoch %d:'%(epoch), recall/ctr, ndcg/ctr)

        ctr, recall, ndcg = 0, 0, 0
        baseline_metrics = {rec_tool:{'recall':0, 'ndcg':0} for rec_tool in test_dataset.rec_tool_order}
        
        for batch_idx, (users, items, test_item, descriptions, test_description, ranking_matrix,_) in tqdm(enumerate(itertools.islice(test_loader, start_batch - 1, start_batch - 1 + k)), total=k, desc="Testing Batches"):
            normalized_ranking_matrix = normalize_list_of_tensors(ranking_matrix)

            batch_ranking = get_ground_truth_ranking(normalized_ranking_matrix, torch.tensor(test_item).to(device))
            batch_weight = user_memory_bank.get_weight_memory_batch(users).to(device)

            aggregate_ranking_score = weighted_sum(batch_weight.to(device),normalized_ranking_matrix)

            batch_user_profiles = user_memory_bank.get_profile_memory_batch(users)
            rerank_positions = get_selected_positions(users, selected_user_ids) if args['selective_rerank'] else list(range(len(users)))
            if args['rerank_infr'] and len(rerank_positions) == len(users):
                topk_values, topk_indices = torch.topk(aggregate_ranking_score, k=args['rerank_infr'], dim=1, largest=True)
                batch_recalled_items = [[(int(idx),train_dataset.item_descriptions[int(idx)]) for idx in row] for row in topk_indices]
                reranked_item_prompt = prompt_bank.get_reranked_item_prompt(batch_user_profiles, descriptions, batch_recalled_items)
                reranked_items = agent.parallel_ranking_evaluation(reranked_item_prompt, mode='rc')
                reranked_items = clean_reranked_list(reranked_items)
                target_item_idx = match_target_item(test_item, reranked_items, args['eval_k'])
                cur_recall, cur_ndcg = compute_metric_at_k_vectorized(target_item_idx, args['eval_k'])
                recall += cur_recall.sum()
                ndcg += cur_ndcg.sum()
                ctr += len(test_item)
            else:
                recall, ndcg, ctr = update_metrics(aggregate_ranking_score, test_item, recall, ndcg, ctr, k=args['eval_k'], device=device)

            baseline_metrics = update_baseline_metrics(baseline_metrics, test_dataset.rec_tool_order, ranking_matrix, test_item, ctr, k=args['eval_k'], device=device)

        recall, ndcg = recall.item()/ctr, ndcg.item()/ctr
        print('Test Result:', recall, ndcg)
        
        local_eval_lr = local_eval_lr * args['lr_decay']
        llm_cmpr_lr = llm_cmpr_lr * args['lr_decay']

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            recall_at_best = recall
            patience_counter = 0

        else:
            patience_counter += 1
            if patience_counter >= args['patience']:
                print("Early stopping triggered.")
                break
    baseline_results = show_baseline_metrics(baseline_metrics, ctr)
    write_log(recall_at_best, best_ndcg, args, baseline_results)

    if args['save_weights']:
        tool_weights = get_tool_weights(start_batch, batch_size, args['num_batch'], user_memory_bank)
        np.save(f'./log/{dataset}/tool_weights.npy',torch.stack(tool_weights).numpy())
