import os, sys
import pandas as pd
import numpy as np
import torch
import csv
from torch.utils.data import Dataset
from tqdm import tqdm 


class RecommendationDataset(Dataset):
    def __init__(self, data_path, valid_path, item_description_path, ranking_score_dict, k = 10):
        self.data = []  # List of (user_id, item_ids)
        self.item_descriptions = {}  # Store item_id -> description mapping
        self.valid_items = {}  # Store user_id -> valid_item_id
        self.ranking_score_dict = ranking_score_dict
        self.rec_tool_order = {}
        self.k = k
        for idx, rec_tool in enumerate(self.ranking_score_dict.keys()):
            self.rec_tool_order[rec_tool] = idx
        # **Load item descriptions from CSV**
        df = pd.read_csv(item_description_path, delimiter='\t', header=0)  # Assuming tab-separated
        for _, row in df.iterrows():
            self.item_descriptions[int(row['new_item_id']) - 1] = row['item_description']

        # **Load valid items from valid.txt**
        num_interaction = 0
        user_set, item_set = set(), set()
        with open(valid_path, 'r') as f:
            for line in f:
                user_id, valid_item = map(int, line.strip().split())
                self.valid_items[user_id] = valid_item  # Store valid item for each user
                user_set.add(user_id)
                item_set.add(valid_item)
                num_interaction += 1


        # **Load user interactions from train.txt**
        with open(data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                user_id = int(parts[0])  # Convert user_id to int
                item_ids = list(map(int, parts[1:]))  # Convert item_ids to list of ints
                self.data.append((user_id, item_ids))
                user_set.add(user_id)
                for item_id in item_ids:
                    item_set.add(item_id)
                    num_interaction += 1
        # print(f'Dataset statistics: num_user = {len(user_set)}, num_item = {len(item_set)}, num_interactions = {num_interaction}, density = {num_interaction/(len(user_set)*len(item_set))}')
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id, item_ids = self.data[idx]

        # **Retrieve valid item for the user (if exists, otherwise use -1)**
        valid_item_id = self.valid_items.get(user_id, -1)

        # **Retrieve item descriptions**
        item_descriptions = [self.item_descriptions.get(item, "Unknown") for item in item_ids]
        valid_item_description = self.item_descriptions.get(valid_item_id, "Unknown")

        ranking_scores, rec_top_item_descriptions = [0] * len(self.ranking_score_dict), [0] * len(self.ranking_score_dict)
        for rec_tool in self.ranking_score_dict.keys():
            rec_item_scores =  self.ranking_score_dict[rec_tool][user_id]
            _, rec_top_item_indices = torch.topk(rec_item_scores, k = self.k)
            rec_top_item_descriptions[self.rec_tool_order[rec_tool]] = [self.item_descriptions[i.item()] for i in rec_top_item_indices]
            ranking_scores[self.rec_tool_order[rec_tool]] = rec_item_scores

        return user_id, item_ids, valid_item_id, item_descriptions, valid_item_description, ranking_scores, rec_top_item_descriptions  # Returning all four components
    
    
def collate_fn(batch):
    user_ids, item_lists, valid_items, desc_lists, valid_desc_lists, ranking_score_list, rec_top_item_descriptions = zip(*batch)  # Unzip batch
    return list(user_ids), list(item_lists), list(valid_items), list(desc_lists), list(valid_desc_lists), list(ranking_score_list), list(rec_top_item_descriptions)  # Keep variable-length lists

def save_user_profile(train_loader, data_path, agent, user_memory_bank, batch_num = None, start_batch = 1):
    """
    Saves all user IDs and batch data from train_loader to a CSV file if the file does not already exist.

    Args:
        train_loader: DataLoader that provides batches of users and related data.
        data_path (str): Path to save the CSV file.
    """
    # Check if the CSV already exists
    if os.path.exists(data_path):
        print(f"File already exists: {data_path}. Skipping save.")
        return

    all_data = []  # List to collect batch data

    processed_batch_num = 0
    for batch_idx, (user_ids, _, _, descriptions, _, _,_) in tqdm(enumerate(train_loader), desc="Processing batches", unit="batch"):
        if batch_idx < start_batch - 1:
            continue
        if batch_num is not None and processed_batch_num >= batch_num:  # Stop after k batches
            break
        
        # Convert descriptions to a list of strings
        user_profiles = agent.generate_user_profile_parallel(descriptions)
        # print(user_profiles)
        user_memory_bank.batch_update(user_ids, string_memories=user_profiles)
        # Store batch data
        for user_id, profile in zip(user_ids, user_profiles):
            all_data.append({"user_id": user_id, "profile": profile})
        processed_batch_num += 1

    # Convert to DataFrame and save
    df = pd.DataFrame(all_data)
    df.to_csv(data_path, index=False, encoding="utf-8")
    print(f"Saved all user batches to {data_path}")

def load_user_profile(profile_path, user_memory_bank):
    with open(profile_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        for row in reader:
            if row[0] == 'user_id': continue
            user_memory_bank.update_memory(int(row[0]), row[1])
    return user_memory_bank
