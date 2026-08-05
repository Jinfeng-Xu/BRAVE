import pandas as pd
import random
import json

def create_item_description_csv(item_info_path, item_map_path, output_csv_path):
    # Load item map (old_item_id -> new_item_id)
    with open(item_map_path, "r") as f:
        item_map = json.load(f)  # Keys are old_item_id (int), values are new_item_id (int)

    print(len(item_map))
    # Load item descriptions (old_item_id -> description)
    with open(item_info_path, "r") as f:
        item_info = json.load(f)  # Keys are old_item_id (str), values are descriptions (str)

    # Convert old_item_id to str to match the JSON format
    item_map_str_keys = {str(k): v for k, v in item_map.items()}
    print(len(item_map_str_keys))

    # Create a list of tuples (new_item_id, item_description)
    mapped_items = [(item_map_str_keys[old_id], desc) for old_id, desc in item_info.items() if old_id in item_map_str_keys]
    print(len(mapped_items))

    # Convert to DataFrame
    df = pd.DataFrame(mapped_items, columns=["new_item_id", "item_description"])
    df = df.sort_values(by="new_item_id")
    # Save as CSV
    df.to_csv(output_csv_path, sep="\t",index=False)

    print(f"New item description CSV saved to {output_csv_path}")

def filter_and_remap_user_item_data(file_path, output_data_path, user_map_path, item_map_path,
                                    min_user_interactions=5, min_item_interactions=5, 
                                    target_users=1000, target_items=10000, srec_datapath = None, columns= ["user_id", "item_id", "timestamp"]):
    # Load data (assumes tab-separated format without header)
    df = pd.read_csv(file_path, sep="\t", names=columns,skiprows=1)
    df["item_id"] = df["item_id"].astype(str)
    df["user_id"] = df["user_id"].astype(str)

    # Step 1: Iteratively filter users and items
    while True:
        prev_shape = df.shape[0]

        # Filter users with at least `min_user_interactions` interactions
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df["user_id"].isin(valid_users)]

        # Filter items with at least `min_item_interactions` users interacting with them
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df["item_id"].isin(valid_items)]

        # Stop if no further filtering occurs
        if df.shape[0] == prev_shape:
            break

    # Step 2: Randomly sample 1000 users if there are more
    unique_users = df["user_id"].unique()
    if len(unique_users) > target_users:
        sampled_users = random.sample(list(unique_users), target_users)
        df = df[df["user_id"].isin(sampled_users)]

        # Step 3: Recheck item interactions after user sampling
        while True:
            prev_shape = df.shape[0]

            # Ensure all items have at least `min_item_interactions` users
            item_counts = df["item_id"].value_counts()
            valid_items = item_counts[item_counts >= min_item_interactions].index
            df = df[df["item_id"].isin(valid_items)]

            # If no more changes, stop
            if df.shape[0] == prev_shape:
                break

    # Step 4: Randomly sample 10,000 items if there are more
    unique_items = df["item_id"].unique()
    if len(unique_items) > target_items:
        sampled_items = random.sample(list(unique_items), target_items)
        df = df[df["item_id"].isin(sampled_items)]

        # Step 5: Recheck user interactions after item sampling
        while True:
            prev_shape = df.shape[0]

            # Ensure all users have at least `min_user_interactions` items
            user_counts = df["user_id"].value_counts()
            valid_users = user_counts[user_counts >= min_user_interactions].index
            df = df[df["user_id"].isin(valid_users)]

            # If no more changes, stop
            if df.shape[0] == prev_shape:
                break

    # Step 6: Remap user and item IDs from 1
    unique_users = df["user_id"].unique()
    unique_items = df["item_id"].unique()

    user_id_map = {old: new_id for new_id, old in enumerate(unique_users, start=1)}
    item_id_map = {old: new_id for new_id, old in enumerate(unique_items, start=1)}

    # Apply remapping
    df["user_id"] = df["user_id"].map(user_id_map)
    df["item_id"] = df["item_id"].map(item_id_map)

    # Step 7: Drop timestamp column and save filtered dataset
    df_filtered = df[["user_id", "item_id"]]
    df_filtered.to_csv(output_data_path, sep="\t", index=False, header=False)
    if srec_datapath:
        df_filtered.to_csv(srec_datapath, sep=" ", index=False, header=False)

    # Step 8: Save user and item mapping dictionaries
    with open(user_map_path, "w") as f:
        json.dump(user_id_map, f)
    
    with open(item_map_path, "w") as f:
        json.dump(item_id_map, f)

    print(f"Filtered data saved to {output_data_path}.")
    print(f"User ID mapping saved to {user_map_path}.")
    print(f"Item ID mapping saved to {item_map_path}.")
    print(f"Final dataset shape: {df_filtered.shape}")
    return user_id_map, item_id_map

# Example usage:
# filter_and_remap_user_item_data("user_item_interactions.txt", "filtered_data.txt", "user_map.json", "item_map.json")
