import torch
import numpy as np
import os
import re
from rapidfuzz import fuzz
from tool_config import embedding_path

def load_embeddings(model_name, dataset, device, embedding_root=None):
    user_path = embedding_path(model_name, dataset, "user", embedding_root=embedding_root)
    item_path = embedding_path(model_name, dataset, "item", embedding_root=embedding_root)
    user_embedding = torch.load(user_path, map_location='cpu', weights_only=True)
    item_embedding = torch.load(item_path, map_location='cpu', weights_only=True)
    return user_embedding.to(device), item_embedding.to(device)

def load_dict(train_file):
    train_dict = {}
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                user_id = int(parts[0])  # First element is the user ID
                item_list = list(map(int, parts[1:]))  # Remaining elements are item IDs
                train_dict[user_id] = item_list
    return train_dict

def load_train_test_sample(dataset):
    if os.path.exists(os.path.join('data', dataset, 'train.txt')):
        save_path = os.path.join('data', dataset)
    else:
        save_path = os.path.join('LightGCN-PyTorch', 'data', dataset)
    train_file = os.path.join(save_path, 'train.txt')
    test_file = os.path.join(save_path, 'test.txt')
    train_dict = load_dict(train_file)
    test_dict = load_dict(test_file)
    return train_dict, test_dict

def mask_train_items(ranking_scores, train_dict):
    """
    Masks the ranking scores of training items by setting them to -inf.
    
    Args:
        ranking_scores (torch.Tensor): Tensor of shape (num_users, num_items), containing ranking scores.
        train_dict (dict): Dictionary {user_id: train_item_list}, where train_item_list contains item indices for training.

    Returns:
        masked_scores (torch.Tensor): Ranking scores with train items set to -inf.
    """
    num_users, num_items = ranking_scores.shape
    device = ranking_scores.device

    # Create a mask initialized with False
    mask = torch.zeros((num_users, num_items), dtype=torch.bool, device=device)

    # Set positions corresponding to training items to True
    for user_id, train_items in train_dict.items():
        mask[user_id, train_items] = True  # Mask train items

    # Apply the mask: Set train item scores to -inf
    ranking_scores.masked_fill_(mask, float('-inf'))

    return ranking_scores

def recall_at_k(ranking_scores, ground_truth, k=10):
    """
    Compute the average Recall@K across all users in a fully vectorized manner.

    Args:
        ranking_scores (torch.Tensor): (num_users, num_items) ranking scores.
        ground_truth (torch.Tensor): (num_users, num_items) binary matrix (1 = relevant, 0 = not relevant).
        k (int): Cutoff for Recall@K.

    Returns:
        float: Average Recall@K across users.
    """
    num_users, num_items = ranking_scores.shape
    device = ranking_scores.device

    # Get top-k item indices for each user
    _, topk_indices = torch.topk(ranking_scores, k, dim=1)  # Shape: (num_users, k)

    # Gather the relevance scores for the top-k items
    topk_relevance = torch.gather(ground_truth, 1, topk_indices)  # Shape: (num_users, k)

    # Compute Recall@K per user
    total_relevant_items = ground_truth.sum(dim=1)  # Total relevant items per user
    total_relevant_items[total_relevant_items == 0] = 1e-10  # Avoid division by zero

    recall_per_user = topk_relevance.sum(dim=1) / total_relevant_items  # Recall per user

    # Return the average Recall@K across users
    return recall_per_user.mean().item()


def dcg_at_k(relevance, k):
    """Compute DCG at rank K in a fully vectorized manner."""
    discounts = torch.log2(torch.arange(1, k + 1, device=relevance.device) + 1)  # log2(i+1)
    return (relevance[:, :k] / discounts).sum(dim=1)  # Sum over top-k items

def ndcg_at_k(ranking_scores, ground_truth, k=10):
    """
    Compute the average NDCG at k across all users in a fully parallelized way using CUDA.
    
    ranking_scores: Tensor of shape (num_users, num_items), ranking scores.
    ground_truth: Tensor of shape (num_users, num_items), binary relevance matrix (1 for relevant, 0 for not relevant).
    k: Cutoff for ranking.
    
    Returns:
        Average NDCG score across users.
    """
    # Move tensors to GPU if available
    device = ranking_scores.device
    
    # Get top-k item indices for each user (shape: num_users x k)
    _, topk_indices = torch.topk(ranking_scores, k, dim=1)

    # Gather relevance scores for the top-k ranked items (num_users x k)
    topk_relevance = torch.gather(ground_truth, 1, topk_indices)

    # Compute DCG for top-k ranked items
    dcg = dcg_at_k(topk_relevance, k)

    # Compute IDCG using the ideal ranking (sorted ground truth relevance per user)
    sorted_ground_truth, _ = torch.sort(ground_truth, descending=True, dim=1)
    idcg = dcg_at_k(sorted_ground_truth, k)

    # Avoid division by zero by setting IDCG=1 where IDCG=0
    idcg[idcg == 0] = 1e-10

    # Compute NDCG and return the average across users
    ndcg_scores = dcg / idcg
    return ndcg_scores.mean().item()

def remove_padding_user(user_emb, item_emb, train_dict):
    if len(train_dict) == user_emb.shape[0] - 1:
        user_emb_trimmed = user_emb[1:]
        item_emb_trimmed = item_emb[1:]
        print('Padding user and item removed')
    else:
        user_emb_trimmed = user_emb
        item_emb_trimmed = item_emb
        print('No padding user and item')
    return user_emb_trimmed, item_emb_trimmed

def convert_test_dict_to_tensor(test_dict, num_users, num_items, device):
    """
    Convert test_dict {user: test_item_list} into a binary ground-truth tensor.
    
    Args:
        test_dict (dict): {user_id: [test_items]}.
        num_users (int): Number of users.
        num_items (int): Number of items.
        device (torch.device): CUDA or CPU.
        
    Returns:
        torch.Tensor: Binary ground truth of shape (num_users, num_items), where 1 means relevant.
    """
    # Initialize a zero matrix (num_users x num_items)
    ground_truth = torch.zeros((num_users, num_items), dtype=torch.float32, device=device)

    # Create user index tensor and item index tensor
    user_ids = torch.tensor(list(test_dict.keys()), device=device)  # User IDs
    item_ids = [torch.tensor(items, device=device) for items in test_dict.values()]  # List of item tensors

    # Concatenate item indices into a single tensor
    item_ids_concat = torch.cat(item_ids)  # All test items in one tensor
    user_ids_repeated = torch.repeat_interleave(user_ids, torch.tensor([len(i) for i in item_ids], device=device))

    # Use scatter_ to mark test items as 1 in the ground truth tensor
    ground_truth[user_ids_repeated, item_ids_concat] = 1.0

    return ground_truth

def normalize_list_of_tensors(batch_list, min_val=-10, max_val=10):
    """
    Normalize a batch of lists of (1 x n) tensors to the range (-10, 10), ignoring `inf` values.

    Args:
        batch_list (list of list of torch.Tensor): Shape [batch_size, 3, n].
        min_val (float): Minimum value of the target range.
        max_val (float): Maximum value of the target range.

    Returns:
        list of list of torch.Tensor: Normalized batch with the same structure.
    """
    # Convert list of lists into a batch_size × 3 × n tensor
    batch_tensor = torch.stack([torch.stack(tensors) for tensors in batch_list])  # Shape: (batch_size, 3, n)

    # Identify valid values while ignoring 'inf'
    valid_mask = ~torch.isinf(batch_tensor)
    valid_values = batch_tensor[valid_mask]
    if valid_values.numel() > 0:  # Ensure there are valid numbers
        min_t, max_t = valid_values.min(), valid_values.max()

        # Avoid division by zero if all values are the same
        if min_t != max_t:
            normalized_values = (valid_values - min_t) / (max_t - min_t) * (max_val - min_val) + min_val
        else:
            normalized_values = torch.full_like(valid_values, (min_val + max_val) / 2)  # Set to midpoint if all values are equal

        # Create a new tensor and place normalized values back
        normalized_tensor = batch_tensor.clone()
        normalized_tensor[valid_mask] = normalized_values
    else:
        normalized_tensor = batch_tensor.clone() # No changes if all values are inf

    return normalized_tensor

def normalize_list_of_tensors_row(batch_list, min_val=-10, max_val=10):
    """
    Normalize each row of a batch of (batch_size, 3, n) tensors to the range (-10, 10), 
    performing row-wise normalization while ignoring `inf` values.

    Args:
        batch_list (list of list of torch.Tensor): Shape [batch_size, 3, n].
        min_val (float): Minimum value of the target range.
        max_val (float): Maximum value of the target range.

    Returns:
        torch.Tensor: Normalized batch with shape (batch_size, 3, n).
    """
    # Convert list of lists into a (batch_size × 3 × n) tensor
    batch_tensor = torch.stack([torch.stack(tensors) for tensors in batch_list])  # Shape: (batch_size, 3, n)

    # Identify valid values while ignoring 'inf'
    valid_mask = ~torch.isinf(batch_tensor)
    
    # Compute row-wise min/max while ignoring 'inf'
    min_t = torch.where(valid_mask, batch_tensor, torch.tensor(float('inf'), device=batch_tensor.device)).min(dim=2, keepdim=True).values
    max_t = torch.where(valid_mask, batch_tensor, torch.tensor(float('-inf'), device=batch_tensor.device)).max(dim=2, keepdim=True).values

    # Avoid division by zero where min_t == max_t
    scale = max_t - min_t
    scale[scale == 0] = 1  # Prevent divide-by-zero errors

    # Proper broadcasting: Normalize using min_t and max_t while preserving shape
    normalized_tensor = batch_tensor.clone()
    normalized_tensor[valid_mask] = ((batch_tensor - min_t) / scale * (max_val - min_val) + min_val)[valid_mask]

    return normalized_tensor

def get_ground_truth_ranking(scores: torch.Tensor, ground_truth_items: torch.Tensor):
    """
    Get the ranking position of each user's ground-truth item across 3 models.

    Args:
        scores (torch.Tensor): Ranking scores of shape (batch_size, 3, item_num)
        ground_truth_items (torch.Tensor): Indices of ground-truth items, shape (batch_size,)
    
    Returns:
        torch.Tensor: Rankings of shape (batch_size, 3)
    """
    # Sort scores along dim=2 (descending order) and get sorted indices
    sorted_indices = torch.argsort(scores, dim=2, descending=True)  # Shape: (batch_size, 3, item_num)

    # Expand ground truth indices to match sorted_indices' shape (batch_size, 3, 1)
    ground_truth_items = ground_truth_items[:, None, None]  # Shape: (batch_size, 1, 1)

    # Compare and find positions where sorted indices match ground-truth items
    ranking_positions = (sorted_indices == ground_truth_items).nonzero(as_tuple=True)[2]  # Extract indices along dim=2

    # Reshape to (batch_size, 3)
    return ranking_positions.view(scores.shape[0], scores.shape[1])

def weighted_sum(weights,batch_ranking_score):
    weights = weights.unsqueeze(-1)  
    weighted_tensor = batch_ranking_score * weights
    output = weighted_tensor.sum(dim=1)
    return output

def compute_batch_recall_ndcg(scores: torch.Tensor, ground_truth_items: torch.Tensor, k: int):
    batch_size, item_num = scores.shape

    top_k_indices = torch.argsort(scores, dim=1, descending=True)[:, :k]  # (batch_size, k)

    hits = (top_k_indices == ground_truth_items.unsqueeze(1))  # (batch_size, k)
    recall_per_user = hits.any(dim=1).float()

    hit_positions = hits.nonzero(as_tuple=False)  # (num_hits, 2)
    dcg_per_user = torch.zeros(batch_size, dtype=torch.float32, device=scores.device)

    if hit_positions.shape[0] > 0:
        user_idx, rank_positions = hit_positions[:, 0], hit_positions[:, 1] + 1
        dcg_per_user.index_add_(0, user_idx, 1.0 / torch.log2(rank_positions.float() + 1))

    # IDCG is always 1.0 since there is only one relevant item ideally at rank 0
    ndcg_per_user = dcg_per_user  # since IDCG = 1.0

    return recall_per_user, ndcg_per_user

def get_batch_rec_tool(rec_tool_dict, batch_size):
    ordered_list = [None] * len(rec_tool_dict)
    for tool, position in rec_tool_dict.items():
        ordered_list[position] = tool
    batch_rec_tool = [ordered_list] * batch_size
    return batch_rec_tool
def get_ranking_tool_pairs(batch_predicted_ranking, batch_rec_tool):
    batch_predicted_ranking = batch_predicted_ranking.tolist()
    batch_predicted_ranking_tool_pairs = [
        list(zip(predicted_ranking, rec_tool)) 
        for predicted_ranking, rec_tool in zip(batch_predicted_ranking, batch_rec_tool)
    ]
    return batch_predicted_ranking_tool_pairs

def get_tool_weights(start_batch, batch_size, num_batch, user_memory_bank):
    start_uid = (start_batch - 1) * 4
    tool_weights = []
    for uid in user_memory_bank.memory:
        if start_uid <= uid < start_uid + batch_size * num_batch:
            tool_weights.append(user_memory_bank.get_memory(uid)[1].cpu())
    return tool_weights

def optimize_by_ranking(rankings, weights, alpha=0.5):
    
    # Inverse ranking normalization (1/rank)
    inv_ranks = 1.0 / rankings
    
    # Normalize to sum to 1
    new_weights = inv_ranks / inv_ranks.sum(dim=1, keepdim=True)
    
    # Smooth update
    updated_weights = alpha * new_weights + (1 - alpha) * weights
    return updated_weights

def map_value(text, mode = 'local', bos = '<BOS>', eos = '<EOS>'):
        text = text.removeprefix(bos).removesuffix(eos)
        if mode == 'local':
            if "IRRELEVANT" in text:
                return -1
            elif "NEUTRAL" in text:
                return 0
            elif "RELEVANT" in text:
                return 1
            else:
                return 0
        elif mode == 'compare':
            match = re.search(r'<BOS>\s*([A-Z])\s*<EOS>', text)
            if match is None:
                match = re.search(r'\b([A-Z])\b', text)
            if match is None:
                return -1
            return ord(match.group(1)) - ord('A')
        elif mode == 2:
            if "GOOD" in text:
                return 2
            elif "GREAT" in text:
                return 1
            elif "POOR" in text:
                return 4
            elif "FAIR" in text:
                return 3
            else:
                return 3

def map_evaluation_to_score(llm_evaluation, weights, device, alpha=0.1):
    mapped_array = np.vectorize(map_value)(llm_evaluation, mode = 2)
    inv_ranks = torch.tensor(1.0 / mapped_array).to(device)
    
    # Normalize to sum to 1
    new_weights = inv_ranks / inv_ranks.sum(dim=1, keepdim=True)
    
    # Smooth update
    updated_weights = alpha * new_weights + (1 - alpha) * weights
    return updated_weights

def optimize_by_local_evaluation(llm_evaluation, weights, device, alpha=0.1):
    mapped_array = np.vectorize(map_value)(llm_evaluation)
    updated_weights = alpha * torch.tensor(mapped_array).to(device) + weights
    return updated_weights

def optimize_by_comparison(llm_evaluation, weights, alpha=0.01):
    best_tool = np.vectorize(map_value)(llm_evaluation, mode = 'compare')
    best_tool = best_tool.flatten()
    for idx, w in enumerate(weights):
        best_tool_idx = best_tool[idx] 
        if best_tool_idx != -1 and best_tool_idx < weights.shape[1]:
            weights[idx][best_tool_idx] = w[best_tool_idx] + alpha
    return weights
        
def update_metrics(aggregate_ranking_score, valid_item, recall, ndcg, ctr, k=10, device="cpu"):
    """
    Compute batch Recall@k and NDCG@k, update cumulative recall, ndcg, and user count.

    Args:
        aggregate_ranking_score (torch.Tensor): Scores for ranking, shape (batch_size, num_items).
        valid_item (list or torch.Tensor): Ground-truth item indices for each user.
        recall (torch.Tensor or float): Running sum of recall scores.
        ndcg (torch.Tensor or float): Running sum of ndcg scores.
        ctr (int): Running count of users.
        k (int, optional): The cutoff value for top-k. Default is 10.
        device (str, optional): The computation device. Default is "cpu".

    Returns:
        tuple: Updated (recall, ndcg, ctr)
    """
    # Compute recall and ndcg for the batch
    batch_recall, batch_ndcg = compute_batch_recall_ndcg(aggregate_ranking_score, torch.tensor(valid_item, device=device), k)
    
    # Update cumulative metrics
    return recall + batch_recall.sum(), ndcg + batch_ndcg.sum(), ctr + len(valid_item)

def create_baseline_metrics(rec_tool_order, batch_size):
    one_hot_dict = {key: torch.eye(len(rec_tool_order))[idx] for key, idx in rec_tool_order.items()}
    batched_one_hot_dict = {key: val.repeat(batch_size, 1) for key, val in one_hot_dict.items()}
    return batched_one_hot_dict

def update_baseline_metrics(baseline_metrics, rec_tool_order, ranking_matrix, test_item, ctr, k, device):
    for rec_tool in baseline_metrics:
        baseline_ranking_score = torch.stack([sublist[rec_tool_order[rec_tool]] for sublist in ranking_matrix])
        baseline_metrics[rec_tool]['recall'], baseline_metrics[rec_tool]['ndcg'], _ = \
        update_metrics(baseline_ranking_score, test_item, baseline_metrics[rec_tool]['recall'], baseline_metrics[rec_tool]['ndcg'], ctr, k=k, device=device)
    return baseline_metrics

def show_baseline_metrics(baseline_metrics, ctr):
    formatted_output = []
    for rec_tool in baseline_metrics:
        baseline_metrics[rec_tool]['recall'] = baseline_metrics[rec_tool]['recall'].item()/ctr
        baseline_metrics[rec_tool]['ndcg'] = baseline_metrics[rec_tool]['ndcg'].item()/ctr
    for key, value in baseline_metrics.items():
        formatted_values = ", ".join(f"{k} = {v:.5f}" for k, v in value.items())
        formatted_output.append(f"{key}: {formatted_values}")
        print(f"{key}: {formatted_values}")
    return formatted_output

def print_log(args):
    args_str = ", ".join([f"{k} = {v}" for k, v in args.items() if k not in ['rec_tool']])
    print(f"Arguments: {args_str}\n")
    return

def slice_nested_list(nested_list, k):
    sliced_list = [
    [
        sublist[:k]
        for sublist in row
    ]
    for row in nested_list
    ]
    return sliced_list

def write_log(best_recall, best_ndcg, args, baseline_results):
    os.makedirs(f"log/{args['dataset']}", exist_ok=True)
    user_name = os.getenv('USER') or os.getenv('USERNAME')
    user_name_suffix = '' if user_name == 'stlamb' else '_'+user_name
    log_path = os.path.join(f"log/{args['dataset']}", f"{args['dataset']}_{args['llm_model']}{user_name_suffix}.txt")
    with open(log_path, "a") as f:
        args_str = ", ".join([f"{k} = {v}" for k, v in args.items() if k not in ['dataset','llm_model', 'rec_tool', 'llm_model_id']])
        f.write(f"Arguments: {args_str}\n")
        f.write(f"\t Baseline Results: {'; '.join(baseline_results)}\n")
        f.write(f"\t Best Results: Recall@{args['top_k']} = {best_recall:.5f}, NDCG@{args['top_k']} = {best_ndcg:.5f}\n")
    return

def write_baseline_log(best_recall, best_ndcg, args, baseline_args: dict):
    os.makedirs(f"log/{args['dataset']}",  exist_ok=True)
    log_path = os.path.join(f"log/{args['dataset']}",  f"{args['dataset']}_{baseline_args['baseline']}.txt")
    with open(log_path, "a") as f:
        args_str = ", ".join([f"{k} = {v}" for k, v in baseline_args.items()])
        f.write(f"Baseline Arguments: {args['llm_model']}, {args_str}\n")
        f.write(f"\t Best Results: Recall@{args['top_k']} = {best_recall:.5f}, NDCG@{args['top_k']} = {best_ndcg:.5f}\n")
    return

def clean_reranked_list(reranked_string_list, separator = '\n'):
    cleaned_list = [s for s in reranked_string_list if s is not None]
    cleaned_list =  [ [x for x in s.split(separator) if x.strip() != '' and re.search(r'\d', x)] for s in cleaned_list]
    return cleaned_list

def first_number_equal(s, valid_item, mode = 'default'):
    """
    Extract the first complete number from a string.

    Args:
        s (str): Input string.

    Returns:
        int | None: The first number if found, otherwise None.
    """
    if mode == 'default':
        return str(valid_item) in s
    else:
        match = re.search(r'\d+', s)
        return int(match.group()) == valid_item

def match_target_item(valid_items, reranked_item_list, top_k):
    top_match_indices = [-1] * len(valid_items)
    for u_idx, (valid_item, reranked_items) in enumerate(zip(valid_items, reranked_item_list)):
        for idx, item in enumerate(reranked_items[:top_k]):
            if first_number_equal(item, valid_item):
                top_match_indices[u_idx] = idx
                break
        if top_match_indices[u_idx] == -1:
            top_match_indices[u_idx] = top_k
    return top_match_indices
        

def predict_top_match_index(target_descriptions, candidate_lists, similarity_func=fuzz.ratio):
    """
    Args:
        target_descriptions: List[str] - N target item descriptions
        candidate_lists: List[List[str]] - N lists of candidate descriptions (e.g., 4x10)
        similarity_func: rapidfuzz similarity function

    Returns:
        List[int] - index of the most similar candidate (0-9) for each target
    """
    top_match_indices = []

    for target, candidates in zip(target_descriptions, candidate_lists):
        similarities = [similarity_func(target, candidate) for candidate in candidates]
        best_index = max(range(len(similarities)), key=lambda i: similarities[i])
        top_match_indices.append(best_index)

    return top_match_indices

def compute_metric_at_k_vectorized(indices, k):
    indices = np.array(indices)
    # Replace -1 or any index >= k with a large value to zero them out later
    valid_mask = (indices >= 0) & (indices < k)
    recall = valid_mask.astype(np.float32)
    ndcg = np.zeros_like(indices, dtype=np.float32)
    ndcg[valid_mask] = 1 / np.log2(indices[valid_mask] + 2)
    return recall, ndcg

def create_dataset_enmf(save_path_train, save_path_test, train_load_path):
    with open(save_path_test, "w") as f:
        f.write("user_id:token\titem_id:token\n")
    with open(save_path_train, "w") as f:
        f.write("user_id:token\titem_id:token\n")
        with open(train_load_path,'r') as f_load:
            for line in f_load:
                line = line.split()
                u, iids = line[0], line[1:]
                for i in iids:
                    f.write(f"{u}\t{i}\n")
    return

def save_test_enmf(save_path_test, user_ids: list, test_item_ids: list):
    with open(save_path_test, "a") as f:
        for u, i in zip(user_ids, test_item_ids):
            f.write(f"{u}\t{i}\n")
    return
