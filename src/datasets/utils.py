import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from typing import Tuple
from pathlib import Path
import vtk

from torch.utils.data import DataLoader

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from .dataset import (
    SequenceDatasetMultiFidelity,
    AutoEncoderDataset
)


def coarse_map_with_convolution(map, coarsing_factor, reduction_method="mode"):
    """
    Coarse a map using convolution to generalize the coarsening process.

    Parameters:
        map (np.array): The input 2D map to coarse.
        coarsing_factor (int): The factor by which the map dimensions will be reduced.
        reduction_method (str): Reduction method to apply ('mode', 'mean', 'max').

    Returns:
        np.array: The coarsened map.
    """

    # Perform coarsening with a sliding window and mode calculation
    coarsened_map = []
    for i in range(0, map.shape[0], coarsing_factor):
        row = []
        for j in range(0, map.shape[1], coarsing_factor):
            subregion = map[i : i + coarsing_factor, j : j + coarsing_factor].flatten()
            if reduction_method == "mode":
                result = stats.mode(subregion).mode
            elif reduction_method == "mean":
                result = np.mean(subregion)
            elif reduction_method == "max":
                result = np.max(subregion)
            row.append(result)
        coarsened_map.append(row)
    return np.array(coarsened_map)


def upscale_map(coarse_map, upscaling_factor, method="nearest"):
    """
    Upscale a coarse map back to a finer resolution using interpolation.

    Parameters:
        coarse_map (np.array): The input 2D coarse map to upscale.
        upscaling_factor (int): The factor by which the coarse map dimensions will be increased.
        method (str): Interpolation method ('nearest', 'bilinear').

    Returns:
        np.array: The upscaled map.
    """
    # Convert the coarse map to a PyTorch tensor
    coarse_map_tensor = torch.tensor(coarse_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Calculate the new size
    new_height = coarse_map.shape[0] * upscaling_factor
    new_width = coarse_map.shape[1] * upscaling_factor

    # Choose the interpolation mode
    if method not in ["nearest", "bilinear"]:
        raise ValueError("Unsupported method. Choose 'nearest' or 'bilinear'.")

    # Perform the upscaling
    upscaled_map_tensor = F.interpolate(
        coarse_map_tensor,
        size=(new_height, new_width),
        mode=method,
        align_corners=False if method == "bilinear" else None,
    )

    # Convert back to numpy array
    upscaled_map = upscaled_map_tensor.squeeze().numpy()

    return upscaled_map


def _encode_data(data, model, shape):
    device = torch.device("cpu")
    data = data.view(-1, *shape)
    with torch.no_grad():
        model = model.to(device)
        # split data in 64 batch size
        model.eval()
        for i in range(0, data.shape[0], 64):
            batch = data[i : i + 64]
            encoded = model.encode(batch.to(device)).cpu().detach()
            if i == 0:
                result = encoded
            else:
                result = torch.cat((result, encoded), dim=0)

        return result


def load_multi_fidelity_dataset_manual_coarsed(
    autoencoder_model,
    data_dir: Path,
    coarse_data_dir: Path,
    val_split: float = 0.1,
    test_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
    batch_size: int = 200,
    slice_time_steps: Tuple[int, int] = None,
    input_len: int = 40,
    output_len: int = 40,
    hop: int = 0,
    real_coarse_data: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    torch.manual_seed(seed)
    data = np.load(data_dir)
    coarse_data = np.load(coarse_data_dir)

    burning_data = data["burning_data"]
    coarse_burning_data = np.zeros_like(burning_data)
    if real_coarse_data:
        cbd = coarse_data["burning_data"]
        for i in range(cbd.shape[0]):
            for j in range(cbd.shape[1]):
                upscaled = upscale_map(cbd[i, j], 8)
                coarse_burning_data[i, j] = upscaled
    else:
        coarse_burning_data = coarse_data["coarse_data"]
    if slice_time_steps:
        burning_data = burning_data[:, slice_time_steps[0] : slice_time_steps[1]]
        coarse_burning_data = coarse_burning_data[:, slice_time_steps[0] : slice_time_steps[1]]
    winds = data["winds"]

    # Extract only the first wind value
    map_wind = lambda x: x[0, 0]
    winds = np.array(list(map(map_wind, winds)))
    winds = winds[:, :2]

    shape = burning_data.shape[2:]
    n_simulations, n_time_steps = burning_data.shape[0], burning_data.shape[1]

    # Use the first n_simulations simulations for training and the rest for testing
    n_val_simulations = int(n_simulations * val_split)
    n_test_simulations = int(n_simulations * test_split)
    n_train_simulations = n_simulations - n_val_simulations - n_test_simulations

    train_data = burning_data[:n_train_simulations]
    val_data = burning_data[n_train_simulations : n_train_simulations + n_val_simulations]
    test_data = burning_data[n_train_simulations + n_val_simulations :]

    coarse_train_data = coarse_burning_data[:n_train_simulations]
    coarse_val_data = coarse_burning_data[
        n_train_simulations : n_train_simulations + n_val_simulations
    ]
    coarse_test_data = coarse_burning_data[n_train_simulations + n_val_simulations :]

    train_data = train_data.reshape(-1, *shape)
    val_data = val_data.reshape(-1, *shape)
    test_data = test_data.reshape(-1, *shape)

    coarse_train_data = coarse_train_data.reshape(-1, *shape)
    coarse_val_data = coarse_val_data.reshape(-1, *shape)
    coarse_test_data = coarse_test_data.reshape(-1, *shape)

    train_data = _encode_data(
        torch.tensor(train_data, dtype=torch.float32), autoencoder_model, shape
    )
    val_data = _encode_data(torch.tensor(val_data, dtype=torch.float32), autoencoder_model, shape)
    test_data = _encode_data(
        torch.tensor(test_data, dtype=torch.float32), autoencoder_model, shape
    )

    coarse_train_data = _encode_data(
        torch.tensor(coarse_train_data, dtype=torch.float32), autoencoder_model, shape
    )
    coarse_val_data = _encode_data(
        torch.tensor(coarse_val_data, dtype=torch.float32), autoencoder_model, shape
    )
    coarse_test_data = _encode_data(
        torch.tensor(coarse_test_data, dtype=torch.float32), autoencoder_model, shape
    )

    latent_dim = train_data.shape[-1]

    train_data = train_data.view(n_train_simulations, -1, latent_dim)
    val_data = val_data.view(n_val_simulations, -1, latent_dim)
    test_data = test_data.view(n_test_simulations, -1, latent_dim)

    coarse_train_data = coarse_train_data.view(n_train_simulations, -1, latent_dim)
    coarse_val_data = coarse_val_data.view(n_val_simulations, -1, latent_dim)
    coarse_test_data = coarse_test_data.view(n_test_simulations, -1, latent_dim)

    train_data = train_data.cpu().detach().numpy()
    val_data = val_data.cpu().detach().numpy()
    test_data = test_data.cpu().detach().numpy()

    coarse_train_data = coarse_train_data.cpu().detach().numpy()
    coarse_val_data = coarse_val_data.cpu().detach().numpy()
    coarse_test_data = coarse_test_data.cpu().detach().numpy()

    train_dataset = SequenceDatasetMultiFidelity(
        train_data,
        coarse_train_data,
        winds,
        input_len,
        output_len,
        hop,
        n_time_steps,
        n_simulations=n_train_simulations,
    )
    val_dataset = SequenceDatasetMultiFidelity(
        val_data,
        coarse_val_data,
        winds,
        input_len,
        output_len,
        hop,
        n_time_steps,
        n_simulations=n_val_simulations,
    )
    test_dataset = SequenceDatasetMultiFidelity(
        test_data,
        coarse_test_data,
        winds,
        input_len,
        output_len,
        hop,
        n_time_steps,
        n_simulations=n_test_simulations,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def load_autoencoder_data(
    data_dir: Path,
    val_split: float = 0.1,
    test_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
    batch_size: int = 200,
    slice_time_steps: Tuple[int, int] = (0, 200),
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    data = np.load(data_dir)

    burning_data = data["burning_data"]
    burning_data = burning_data[:, slice_time_steps[0] : slice_time_steps[1]]
    shape = burning_data.shape[2:]
    n_simulations, n_time_steps = burning_data.shape[0], burning_data.shape[1]

    # Use the first n_simulations simulations for training and the rest for testing
    n_val_simulations = int(n_simulations * val_split)
    n_test_simulations = int(n_simulations * test_split)
    n_train_simulations = n_simulations - n_val_simulations - n_test_simulations

    train_data = burning_data[:n_train_simulations]
    val_data = burning_data[n_train_simulations : n_train_simulations + n_val_simulations]
    test_data = burning_data[n_train_simulations + n_val_simulations :]

    train_data = train_data.reshape(-1, *shape)
    val_data = val_data.reshape(-1, *shape)
    test_data = test_data.reshape(-1, *shape)

    train_dataset = AutoEncoderDataset(train_data, n_time_steps, n_simulations=n_train_simulations)
    val_dataset = AutoEncoderDataset(val_data, n_time_steps, n_simulations=n_val_simulations)
    test_dataset = AutoEncoderDataset(test_data, n_time_steps, n_simulations=n_test_simulations)

    torch.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=7, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=7, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=7, persistent_workers=True)

    return train_dataloader, val_dataloader, test_dataloader

from vtk.util import numpy_support


def vtk_to_numpy(
        starting_timestep: int = 20,
        ending_timestep: int = 8020,
        timestep_interval: int = 20,
        number_of_files: int = 50,
        folder_path: str = "",
        verbose: bool = False,
        numpy_output_folder: str = "",
        shape: tuple = (100, 100)
):
    timesteps = range(starting_timestep, ending_timestep + timestep_interval, timestep_interval)
    reader = vtk.vtkStructuredPointsReader()
    files = range(0, number_of_files)
    burning_data = []

    burning_data = []
    for j in files:
        if verbose:
            print("Managing file", j)
        b_data = []
        for i in timesteps:
            reader.SetFileName(f"{folder_path}/sim{j}/VTK/Solution.vtk.{i}")
            reader.Update()
            data = reader.GetOutput()
            point_data = data.GetPointData()
            point_data_arrays = []
            for i in range(point_data.GetNumberOfArrays()):
                point_data_arrays.append(numpy_support.vtk_to_numpy(point_data.GetArray(i)))
            point_data_arrays = np.array(point_data_arrays)
            point_data_arrays = np.reshape(point_data_arrays, shape)
            b_data.append(point_data_arrays)
        burning_data.append(b_data)

    elevations = []
    for j in files:
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(f"{folder_path}/sim{j}/VTK/Elevation.vtk")
        reader.Update()

        data = reader.GetOutput()
        point_data = data.GetPointData()
        for i in range(point_data.GetNumberOfArrays()):
            elevation = numpy_support.vtk_to_numpy(point_data.GetArray(0))
        elevation = np.reshape(elevation, shape)
        elevations.append(elevation)

    # import wind vtk data
    winds = []
    for j in files:
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(f"{folder_path}/sim{j}/VTK/Wind.vtk")
        reader.Update()

        data = reader.GetOutput()
        point_data = data.GetPointData()
        wind = numpy_support.vtk_to_numpy(point_data.GetArray(0))
        wind = wind.reshape(*shape, 3)
        winds.append(wind)

    # import landuse vtk data
    landuses = []
    for j in files:
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(f"{folder_path}/sim{j}/VTK/Landuse.vtk")
        reader.Update()

        data = reader.GetOutput()
        point_data = data.GetPointData()
        landuse = numpy_support.vtk_to_numpy(point_data.GetArray(0))
        landuse = landuse.reshape(shape)
        landuses.append(landuse)

    np.savez(numpy_output_folder, burning_data=burning_data, elevations=elevations, winds=winds, landuses=landuses)
