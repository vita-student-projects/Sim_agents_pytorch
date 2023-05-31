import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, pickle_files):
        self.pickle_files = pickle_files

    def __len__(self):
        return len(self.pickle_files)

    def __getitem__(self, idx):
        pickle_file = self.pickle_files[idx]
        processed = pickle.load(open(pickle_file, 'rb'))
        
        trajectories = torch.Tensor(processed['track_infos']['trajs'])
        tracks_until_current = torch.Tensor(processed['track_infos']['tracks_to_predict_until_current'])
        tracks_future = torch.Tensor(processed['track_infos']['tracks_to_predict_future'])
        adv_until_current = torch.Tensor(processed['track_infos']['track_of_adv_until_current'])
        adv_future = torch.Tensor(processed['track_infos']['track_of_adv_future'])
        other_agents_until_current = torch.Tensor(processed['track_infos']['tracks_of_other_agents_until_current'])
        other_agents_future = torch.Tensor(processed['track_infos']['tracks_of_other_agents_future'])
        
        return (
            trajectories, tracks_until_current, tracks_future,
            adv_until_current, adv_future,
            other_agents_until_current, other_agents_future
        )

def collate_fn(batch):
    return batch

def create_dataloader(directory, batch_size):
    pickle_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.pkl')]
    dataset = MyDataset(pickle_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader