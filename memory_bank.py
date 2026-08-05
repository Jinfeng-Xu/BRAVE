import torch
class UserMemoryBank:
    def __init__(self, num_user, dataset, rec_tools, initial_weight = None):
        """
        Initialize the memory bank with a default string and list for each user.
        
        Args:
            user_ids (list or tensor): List of user IDs that the memory bank should track.
        """

        weight_dim = len(rec_tools)
        if initial_weight == None:
            if dataset == 'amazon' and weight_dim == 3:
                initial_weight = torch.tensor([0.2,0.4,0.4])
            else:
                initial_weight = torch.full((weight_dim,),1.0 / weight_dim)
        else:
            initial_weight = torch.tensor(initial_weight)

        if dataset == 'amazon':
            initial_profile = 'This user is interested in movies on Amazon.'
        elif dataset == 'yelp':
            initial_profile = 'This user is interested in businesses on Yelp.'
        elif dataset == 'goodreads':
            initial_profile = 'This user is interested in books on Goodreads.'
        else:
            initial_profile = 'This user has historical interactions in the recommendation dataset.'
        
        self.memory = {
            user_id: {
                "profile_memory": initial_profile, 
                "weight_memory": initial_weight,
                } 
                for user_id in range(num_user)
        }

    def update_memory(self, user_id, profile_memory=None, weight_memory=None):
        """
        Update the memory of a given user.
        
        Args:
            user_id (int): The ID of the user whose memory should be updated.
            profile_memory (str, optional): New string memory to store (overwrites).
            weight_memory (list, optional): New list memory to store (overwrites).
        """
        if user_id not in self.memory:
            raise ValueError(f"User ID {user_id} is not in the memory bank.")

        if profile_memory is not None:
            self.memory[user_id]["profile_memory"] = profile_memory

        if weight_memory is not None:
            if weight_memory.shape != self.memory[user_id]["weight_memory"].shape:
                raise ValueError(f"Shape mismatch for weight memory of user {user_id}. Expected {self.memory[user_id]['weight_memory'].shape}, got {weight_memory.shape}.")
            self.memory[user_id]["weight_memory"] = weight_memory

    def get_weight_memory_batch(self, user_ids):
        """
        Retrieve weight memories for a batch of users.

        Args:
            user_ids (list or tensor): List of user IDs.

        Returns:
            Tensor: A batch of weight memories (batch_size, weight_dim).
        """
        weight_memory_batch = torch.stack([self.memory[user_id]["weight_memory"] for user_id in user_ids])
        return weight_memory_batch  # Shape: (batch_size, weight_dim)
    
    def get_profile_memory_batch(self, user_ids):
        """
        Retrieve weight memories for a batch of users.

        Args:
            user_ids (list or tensor): List of user IDs.

        Returns:
            Tensor: A batch of weight memories (batch_size, weight_dim).
        """
        profile_memory_batch = [self.memory[user_id]["profile_memory"] for user_id in user_ids]
        return profile_memory_batch 
    
    def get_memory(self, user_id):
        """
        Retrieve the memory of a given user.
        
        Args:
            user_id (int): The ID of the user to retrieve memory for.
        
        Returns:
            tuple: (profile_memory, weight_memory)
        """
        if user_id not in self.memory:
            raise ValueError(f"User ID {user_id} is not in the memory bank.")
        return self.memory[user_id]["profile_memory"], self.memory[user_id]["weight_memory"]

    def batch_update(self, user_ids, string_memories=None, weights_memories=None):
        """
        Update memories for multiple users at once.
        
        Args:
            user_ids (list): List of user IDs to update.
            string_memories (list, optional): List of strings to store (length must match user_ids).
            list_memories (list, optional): List of lists to store (length must match user_ids).
        """
        for i, user_id in enumerate(user_ids):
            self.update_memory(
                user_id,
                profile_memory=string_memories[i] if string_memories else None,
                weight_memory = weights_memories[i] if weights_memories is not None and len(weights_memories) > 0 else None
            )
