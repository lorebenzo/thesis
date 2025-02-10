import numpy as np
import torch
from torch.utils.data import Dataset



class AutoEncoderDataset(Dataset):
    def __init__(self, data, n_samples=400, n_simulations=50):
        self.data = data
        self.__version__ = "1.0.0"

        # data is a dictionary with burning_data and winds
        self.n_samples = n_samples
        self.n_simulations = n_simulations

    def __len__(self):
        return self.n_samples * self.n_simulations

    def __getitem__(self, idx):
        burning_data = self.data[idx]

        result = torch.tensor(burning_data, dtype=torch.float32)
        return result, result


class SequenceDatasetMultiFidelity(Dataset):
    def __init__(
        self, data, coarse_data, winds, input_len, output_len, hop, n_samples=200, n_simulations=50
    ):
        """
        data: A 2D array of shape (num_timesteps, num_features) where
              num_features is 20 in your case.
        input_len: The length of the input sequence (e.g., 10).
        output_len: The length of the output sequence (e.g., 15).
        """
        self.data = data
        self.coarse_data = coarse_data
        self.input_len = input_len
        self.output_len = output_len
        self.hop = hop
        self.n_samples = n_samples
        self.n_simulations = n_simulations
        self.winds = winds

    def __len__(self):
        # data is number of simulations x timesteps x features, so we need to subtract the input and output lengths
        # return (self.n_samples - self.input_len - self.output_len + 1) * self.n_simulations
        # hop is the number of timesteps we skip between input and output sequences
        return (self.n_samples - self.input_len - self.output_len - self.hop) * self.n_simulations

    def __getitem__(self, idx):
        # the idx is the index from which we calculate the time step and the simulation number
        sim_num = idx // (self.n_samples - self.input_len - self.output_len - self.hop)
        t = idx % (self.n_samples - self.input_len - self.output_len - self.hop)

        # get the input and output sequences
        input_seq = self.data[sim_num, t : t + self.input_len]
        output_seq = self.data[
            sim_num,
            t + self.input_len + self.hop : t + self.input_len + self.output_len + self.hop,
        ]

        coarse_input_seq = self.coarse_data[sim_num, t : t + self.input_len]
        coarse_output_seq = self.coarse_data[
            sim_num,
            t + self.input_len + self.hop : t + self.input_len + self.output_len + self.hop,
        ]

        # concat the input sequence and the wind data (multiply the wind data for each cell)

        wind_data = self.winds[sim_num]
        wind_data = np.tile(wind_data, (input_seq.shape[0], 1))

        input_seq = np.concatenate((input_seq, wind_data), axis=1)
        coarse_input_seq = np.concatenate((coarse_input_seq, wind_data), axis=1)

        input_seq = torch.tensor(input_seq, dtype=torch.float32)
        coarse_input_seq = torch.tensor(coarse_input_seq, dtype=torch.float32)
        output_seq = torch.tensor(output_seq, dtype=torch.float32)
        output_coarse_seq = torch.tensor(coarse_output_seq, dtype=torch.float32)

        return (input_seq, coarse_input_seq), (output_seq, output_coarse_seq)

class SequenceDatasetNeuralODE(Dataset):
    def __init__(
        self, data, winds, output_len, n_samples=200, n_simulations=50
    ):
        """
        data: A 2D array of shape (num_timesteps, num_features) where
              num_features is 20 in your case.
        input_len: The length of the input sequence (e.g., 10).
        output_len: The length of the output sequence (e.g., 15).
        """
        self.data = data
        self.output_len = output_len
        self.n_samples = n_samples
        self.n_simulations = n_simulations
        self.winds = winds

    def __len__(self):
        return (self.n_samples - self.output_len) * self.n_simulations

    def __getitem__(self, idx):
        # the idx is the index from which we calculate the time step and the simulation number
        sim_num = idx // (self.n_samples - self.output_len)
        t = idx % (self.n_samples - self.output_len)

        x = self.data[sim_num, t : t + self.output_len]
        wind_data = self.winds[sim_num]
        wind_data = np.tile(wind_data, (x.shape[0], 1))
        x = np.concatenate((x, wind_data), axis=1)
        x = torch.tensor(x, dtype=torch.float32)
        return x, x