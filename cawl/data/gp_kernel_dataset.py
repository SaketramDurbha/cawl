import numpy as np
import torch
from torch.utils.data import Dataset


def extract_data(
    dataset: "GPKernelShipClassificationDataset",
) -> tuple[np.ndarray, np.ndarray, list]:
    """Extract features, labels, and MMSIs from the dataset."""
    features = []
    labels = []
    mmsis = []

    for i in range(len(dataset)):
        mmsi, kernel_params, group_id = dataset[i]
        features.append(kernel_params.cpu().numpy())
        labels.append(group_id)
        mmsis.append(mmsi)

    return np.array(features), np.array(labels), mmsis


class GPKernelShipClassificationDataset(Dataset):
    """
    Dataset for ship classification using Gaussian Process kernel parameters. Takes fitted GP models and extracts kernel parameters for each ship (identified by MMSI).
    These kernel parameters are used for classification tasks, where each ship is classified into a group based on its kernel parameters.
    Note that the group_id is distinct from the AIS vessel code. The map can be found in the gp_regression_dataset or at https://coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf
    The dataset can also handle unscaled kernel parameters if scalers are provided for each MMSI.
    """

    def __init__(self, gp_regression_dataset, models, device, scalers_by_mmsi=None):
        """_summary_

        Args:
            gp_regression_dataset (_type_): Dataset containing GP regression data. (Used to extract MMSI and class labels)
            models (_type_): Dictionary of fitted GP models per MMSI, where keys are MMSI and values are GP models.
            device (_type_): _description_
        """
        self.gp_regression_dataset = gp_regression_dataset
        self.models = models
        self.device = device
        self.scalers_by_mmsi = scalers_by_mmsi

        # self.kernel_params = kernel_params # Dictionary of fitted GP kernel parameters per MMSI
        self.data = []  # List of (mmsi, kernel_params_tensor, group_id) tuples
        self.mmsis = []  # List of MMSI identifiers

        gp_kernel_ship_classification = {}

        for mmsi in models:
            model = models[mmsi]
            kernel_params = self.extract_kernel_params(model, mmsi)
            group_id = gp_regression_dataset.get_vessel_group_id_by_mmsi(mmsi)
            gp_kernel_ship_classification[mmsi] = {
                "kernel_params": kernel_params,
                "group_id": group_id,
            }

            kernel_params_tensor = torch.tensor(
                list(kernel_params.values()), dtype=torch.float32
            )
            self.data.append((mmsi, kernel_params_tensor, group_id))
            self.mmsis.append(mmsi)

    def extract_kernel_params(self, model, mmsi=None):
        params = {}
        kernels = model.covar_module.data_covar_module.kernels

        # Add all model parameters
        for param_name, param in model.named_parameters():
            # Handle different parameter shapes
            if param.numel() == 1:  # Single value parameter
                params[f"param_{param_name.replace('.', '_')}"] = param.item()
            else:  # Multi-dimensional parameter
                # For vectors, we can add each element separately
                if param.dim() == 1:
                    for i, val in enumerate(param.tolist()):
                        params[f"param_{param_name.replace('.', '_')}_{i}"] = float(val)
                # Handle 2D tensors - flatten them first
                else:
                    flat_values = param.flatten().tolist()
                    for i, val in enumerate(flat_values):
                        params[f"param_{param_name.replace('.', '_')}_{i}"] = float(val)
                    # For larger tensors, we can use statistics
                    params[f"param_{param_name.replace('.', '_')}_mean"] = (
                        param.mean().item()
                    )
                    params[f"param_{param_name.replace('.', '_')}_std"] = (
                        param.std().item()
                    )
                    params[f"param_{param_name.replace('.', '_')}_min"] = (
                        param.min().item()
                    )
                    params[f"param_{param_name.replace('.', '_')}_max"] = (
                        param.max().item()
                    )

        # Add standardized scalar info if available
        if (
            mmsi is not None
            and self.scalers_by_mmsi is not None
            and mmsi in self.scalers_by_mmsi
        ):
            scaler_dict = self.scalers_by_mmsi[mmsi]

            # Add time scaler statistics
            if "time_scaler" in scaler_dict:
                time_scaler = scaler_dict["time_scaler"]
                params["time_scale_factor"] = time_scaler.scale_[0]
                params["time_mean"] = time_scaler.mean_[0]
                params["time_var"] = time_scaler.var_[0]

            # Add state scaler statistics
            if "state_scaler" in scaler_dict:
                state_scaler = scaler_dict["state_scaler"]

                # For multivariate state, add statistics for each dimension
                for i, (scale, mean, var) in enumerate(
                    zip(state_scaler.scale_, state_scaler.mean_, state_scaler.var_)
                ):
                    params[f"state_{i}_scale_factor"] = scale
                    params[f"state_{i}_mean"] = mean
                    params[f"state_{i}_var"] = var

                # Add aggregate statistics
                params["state_scale_factor_mean"] = np.mean(state_scaler.scale_)
                params["state_scale_factor_std"] = np.std(state_scaler.scale_)
                params["state_mean_mean"] = np.mean(state_scaler.mean_)
                params["state_mean_std"] = np.std(state_scaler.mean_)

        # print("Extracted parameter names for MMSI", mmsi, ":", len(list(params.keys())))

        return params

    def get_parameter_names(self):
        """
        Returns a list of parameter names that are extracted from each model.

        Returns:
            list: List of parameter names
        """
        if not hasattr(self, "_param_names") or self._param_names is None:
            # Generate parameter names by extracting from a model
            if self.models:
                first_model = next(iter(self.models.values()))
                params = self.extract_kernel_params(first_model)
                self._param_names = list(params.keys())
            else:
                self._param_names = []

        return self._param_names

    def get_unique_group_ids(self):
        """
        Returns a list of unique group_ids in the dataset.
        """
        return list(set(group_id for _, _, group_id in self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mmsi, kernel_params, group_id = self.data[idx]

        # If scalers are provided, unscale all kernel parameters
        if self.scalers_by_mmsi is not None and mmsi in self.scalers_by_mmsi:
            scaler_dict = self.scalers_by_mmsi[mmsi]
            param_names = self.get_parameter_names()
            kernel_params_unscaled = []

            for i, param_name in enumerate(param_names):
                param_value = kernel_params[
                    i
                ].item()  # GPKernelShipClassificationNetworkkernel params

                # Apply appropriate unscaling based on parameter type
                if "lengthscale" in param_name and "time_scaler" in scaler_dict:
                    # Time-related parameters
                    time_scaler = scaler_dict["time_scaler"]
                    unscaled_value = param_value * time_scaler.scale_[0]
                elif "variance" in param_name and "state_scaler" in scaler_dict:
                    # Variance parameters
                    state_scaler = scaler_dict["state_scaler"]
                    unscaled_value = param_value * (state_scaler.scale_[0] ** 2)
                elif (
                    any(x in param_name for x in ["std", "mean", "min", "max"])
                    and "state_scaler" in scaler_dict
                ):
                    # Statistical parameters from state
                    state_scaler = scaler_dict["state_scaler"]
                    unscaled_value = param_value * state_scaler.scale_[0]
                else:
                    # Default: no unscaling for parameters we don't know how to unscale
                    unscaled_value = param_value

                kernel_params_unscaled.append(unscaled_value)

            # Convert back to tensor
            kernel_params_unscaled = torch.tensor(
                kernel_params_unscaled,
                dtype=kernel_params.dtype,
                device=kernel_params.device,
            )
        else:
            kernel_params_unscaled = kernel_params

        return mmsi, kernel_params_unscaled.to(self.device), group_id

        return mmsi, kernel_params, group_id
