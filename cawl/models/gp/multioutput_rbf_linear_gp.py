import os
import re
import time
from datetime import datetime

import gpytorch
import pandas as pd
import seaborn as sns
import torch
from control import *
from IPython.display import clear_output
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sns.set_theme(style="whitegrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiOutputExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_outputs):
        super(MultiOutputExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_outputs = num_outputs

        ### Define a mean module for multitask GPs ###
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_outputs
        )

        ### Define a covariance module for multitask GPs ###
        # self.covar_module = gpytorch.kernels.MultitaskKernel(
        #     gpytorch.kernels.ScaleKernel(
        #         gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel()
        #     ),
        # num_tasks=num_outputs, rank=1
        # )

        ## Multitask Kernel
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel(),
            num_tasks=num_outputs,
            rank=1,
        )

        ## Independent Multitask Kernel
        # self.covar_module = gpytorch.kernels.IndependentMultitaskKernel(
        #     gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel(),
        #     num_tasks=num_outputs,
        #     rank=1
        # )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def train_model(
    model,
    likelihood,
    train_x,
    train_y,
    num_epochs=500,
    lr=0.01,
    mmsi=None,
    session_id=None,
):
    # Log the training loss to TensorBoard
    # if mmsi is None:
    #     log_dir = f"logs/gp_regression/{mmsi}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # else:
    #     log_dir = f"logs/gp_regression/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # writer = SummaryWriter(log_dir=log_dir)

    writer = None
    if session_id:
        log_dir = f"logs/gp_regression/{session_id}"
        if mmsi:
            log_dir = f"{log_dir}/mmsi_{mmsi}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    model.to(device)
    likelihood.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # for i in tqdm(range(num_epochs), desc=f"GP Training Progress"):
    for i in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        tag_prefix = f"{mmsi}/" if mmsi is not None else ""
        writer.add_scalar(f"Training Loss/{tag_prefix}", loss.item(), i)
        writer.add_scalar(
            f"Length Scale/{tag_prefix}",
            model.covar_module.data_covar_module.kernels[0].lengthscale.item(),
            i,
        )
        writer.add_scalar(
            f"Variance/{tag_prefix}",
            model.covar_module.data_covar_module.kernels[1].variance.item(),
            i,
        )

    writer.flush()
    writer.close()
    # print(f"Training completed. Loss: {loss.item()}")

    return loss, model, likelihood


def eval_model(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(
        1e-3
    ):
        observed_pred = likelihood(model(test_x))

        return observed_pred


def plot_gp(train_x, train_y, test_x, observed_pred):
    # Define a colormap for each state
    colormap = sns.color_palette("colorblind", 6)

    with torch.no_grad():
        # Initialize plot with 2x3 subplots
        f, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()

        # Labels for the DoFs
        dof_labels = [
            "x (m)",
            "y (m)",
            r"$\theta$ (rad)",
            r"$\dot{x}$ (m/s)",
            r"$\dot{y}$ (m/s)",
            r"$\dot{\theta}$ (rad/s)",
        ]

        # Plot predictive means and confidence bounds for DoFs 1, 2, and 3 in the first row
        for i in range(3):
            ax = axes[0, i]
            # Plot training data as black stars
            ax.scatter(
                train_x.cpu().numpy().flatten(),
                train_y.cpu().numpy()[:, i],
                color=colormap[i],
                marker="x",
            )
            # Plot predictive means
            ax.plot(
                test_x.cpu().numpy().flatten(),
                observed_pred.mean[:, i].cpu().numpy(),
                color=colormap[i],
            )
            # Plot confidence bounds
            lower_bound = lower[:, i].cpu().numpy()
            upper_bound = upper[:, i].cpu().numpy()
            if i == 0 or i == 1:
                lower_bound /= 1.1
                upper_bound *= 1.1
            ax.fill_between(
                test_x.cpu().numpy().flatten(),
                lower_bound,
                upper_bound,
                color=colormap[i],
                alpha=0.2,
            )
            ax.set_ylabel(dof_labels[i])

        # Plot predictive means and confidence bounds for DoFs 4, 5, and 6 in the second row
        for i in range(3, 6):
            ax = axes[1, i - 3]
            # Plot training data as black stars
            ax.scatter(
                train_x.cpu().numpy().flatten(),
                train_y.cpu().numpy()[:, i],
                color=colormap[i],
                marker="x",
            )
            # Plot predictive means
            ax.plot(
                test_x.cpu().numpy().flatten(),
                observed_pred.mean[:, i].cpu().numpy(),
                color=colormap[i],
            )
            # Plot confidence bounds
            lower_bound = lower[:, i].cpu().numpy()
            upper_bound = upper[:, i].cpu().numpy()
            if i == 0 or i == 1:
                lower_bound /= 1.1
                upper_bound *= 1.1

            ax.fill_between(
                test_x.cpu().numpy().flatten(),
                lower_bound,
                upper_bound,
                color=colormap[i],
                alpha=0.2,
            )
            ax.set_ylabel(dof_labels[i])

        # Set common x-label
        for ax in axes[-1, :]:
            ax.set_xlabel("Time (s)")

        # Create a single legend
        # legend_elements = [
        #     plt.Line2D([0], [0], color=colormap[i], lw=2, label=f'DoF {dof_labels[i]}') for i in range(6)
        # ]
        legend_elements = []
        legend_elements.append(
            plt.Line2D([0], [0], color="black", marker=".", label="Observed Data")
        )
        legend_elements.append(
            plt.Line2D([0], [0], color="black", linestyle="-", label="Predictive Mean")
        )
        f.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 1.05),
        )
        # Increase font size of axis labels
        for ax in axes.flat:
            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)
        plt.tight_layout()
        plt.show()


###


def save_model_bundle(models, likelihoods, losses, date_range, save_path):
    """Save a bundle of GP models to disk."""
    save_dict = {
        "models_state_dict": {k: v.state_dict() for k, v in models.items()},
        "likelihoods_state_dict": {k: v.state_dict() for k, v in likelihoods.items()},
        "losses": losses,
        "date_range": [
            date_range[0].strftime("%Y-%m-%d"),
            date_range[-1].strftime("%Y-%m-%d"),
        ],
        "num_trajectories": len(models),
    }

    torch.save(save_dict, save_path)
    print(f"Saved model bundle to {save_path}")


def load_exact_model_bundle(save_path, gp_regression_dataset, device):
    """Load models from a bundle that exactly matches the requested parameters."""
    models = {}
    likelihoods = {}
    losses = {}

    bundle = torch.load(save_path, map_location=device)
    models_state_dict = bundle["models_state_dict"]
    likelihoods_state_dict = bundle["likelihoods_state_dict"]
    losses = bundle.get("losses", {})

    # Verify the model count matches expected count
    if len(models_state_dict) != len(gp_regression_dataset):
        print(
            f"Warning: Loaded {len(models_state_dict)} models but expected {len(gp_regression_dataset)}"
        )

    # Reconstruct models from state dictionaries
    for mmsi, state_dict in tqdm(models_state_dict.items(), desc="Loading models"):
        # Get the corresponding data for this MMSI
        times, state_trajectory = None, None
        for entry in gp_regression_dataset:
            if entry[0] == mmsi:
                _, times, state_trajectory = entry
                break
        if times is None:
            print(f"Warning: Could not find data for MMSI {mmsi}, skipping")
            continue

        X = times.detach().unsqueeze(1).to(device)
        Y = state_trajectory.detach().to(device)
        num_outputs = Y.shape[1]

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_outputs
        ).to(device)

        model = MultiOutputExactGPModel(X, Y, likelihood, num_outputs=num_outputs).to(
            device
        )

        # Load the state dictionaries
        model.load_state_dict(state_dict)
        if mmsi in likelihoods_state_dict:
            likelihood.load_state_dict(likelihoods_state_dict[mmsi])

        models[mmsi] = model
        likelihoods[mmsi] = likelihood

    print(f"Successfully loaded {len(models)} models")
    return models, likelihoods, losses


def find_partial_bundles(save_dir, date_range):
    """Find model bundles that partially overlap with the requested date range."""
    partial_bundles = []
    # start_date = date_range[0].strftime('%Y-%m-%d')
    # end_date = date_range[-1].strftime('%Y-%m-%d')

    start_date = date_range[0].strftime("%Y-%m-%d") if len(date_range) > 0 else None
    end_date = date_range[-1].strftime("%Y-%m-%d") if len(date_range) > 0 else None

    # Find model bundles for individual days or smaller ranges within our target range
    for filename in os.listdir(save_dir):
        if not filename.startswith("rbf_and_linear_kernel_classification_bundle_"):
            continue

        # Extract date information from filename
        match = re.search(
            r"bundle_(\d{4}-\d{2}-\d{2})(?:_to_(\d{4}-\d{2}-\d{2}))?_n(\d+)", filename
        )
        if not match:
            continue

        bundle_start = match.group(1)
        bundle_end = match.group(2) if match.group(2) else bundle_start
        bundle_count = int(match.group(3))

        # Convert to datetime for comparison
        bundle_start_date = pd.to_datetime(bundle_start)
        bundle_end_date = pd.to_datetime(bundle_end)

        # Check if any part of this bundle overlaps with our requested date range
        requested_start = pd.to_datetime(start_date)
        requested_end = pd.to_datetime(end_date)

        # Check for overlap
        if bundle_start_date <= requested_end and bundle_end_date >= requested_start:
            # There's overlap, add this bundle to our list
            bundle_dates = pd.date_range(start=bundle_start, end=bundle_end, freq="D")
            partial_bundles.append(
                (bundle_dates, os.path.join(save_dir, filename), bundle_count)
            )

    return partial_bundles


def load_prioritized_bundles(partial_bundles, gp_regression_dataset, device):
    """Load models from partial bundles with prioritization."""
    models = {}
    likelihoods = {}
    losses = {}

    previously_loaded_count = 0
    loaded_mmsis_by_date = {}

    # Extract date information from regression dataset if available
    dates_in_dataset = []
    for entry in gp_regression_dataset:
        if len(entry) > 3 and entry[3] is not None:
            dates_in_dataset.append(entry[3])

    # Only create date range if we have dates
    if dates_in_dataset:
        min_date = min(dates_in_dataset)
        max_date = max(dates_in_dataset)
        date_range = pd.date_range(start=min_date, end=max_date, freq="D")
        loaded_mmsis_by_date = {date: set() for date in date_range}

    # Load models from prioritized bundles
    for bundle_dates, bundle_path, bundle_count in partial_bundles:

        bundle_name = os.path.basename(bundle_path)
        print(f"Loading models from {bundle_name}")

        bundle = torch.load(bundle_path)
        models_state_dict = bundle["models_state_dict"]
        likelihoods_state_dict = bundle.get("likelihoods_state_dict", {})
        bundle_losses = bundle.get("losses", {})

        # Load models
        for mmsi, state_dict in tqdm(
            models_state_dict.items(), desc=f"Loading from bundle"
        ):
            # Skip if we already loaded this MMSI (from a larger bundle)
            if mmsi in models:
                continue

            # Get the corresponding data for this MMSI
            times, state_trajectory, date_for_mmsi = None, None, None
            for entry in gp_regression_dataset:
                if entry[0] == mmsi:
                    _, times, state_trajectory = entry
                    # Extract the date for this MMSI
                    date_for_mmsi = entry[3] if len(entry) > 3 else None
                    break

            if times is None:
                continue  # Skip if MMSI not found

            # Skip if we already loaded this MMSI for this date
            if date_for_mmsi and mmsi in loaded_mmsis_by_date.get(date_for_mmsi, set()):
                continue

            X = times.detach().unsqueeze(1).to(device)
            Y = state_trajectory.detach().to(device)
            num_outputs = Y.shape[1]

            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=num_outputs
            ).to(device)

            model = MultiOutputExactGPModel(
                X, Y, likelihood, num_outputs=num_outputs
            ).to(device)

            # Load the state dictionaries
            model.load_state_dict(state_dict)
            if mmsi in likelihoods_state_dict:
                likelihood.load_state_dict(likelihoods_state_dict[mmsi])

            models[mmsi] = model
            likelihoods[mmsi] = likelihood
            if mmsi in bundle_losses:
                losses[mmsi] = bundle_losses[mmsi]

            # Mark this MMSI as loaded for its date
            if date_for_mmsi:
                loaded_mmsis_by_date[date_for_mmsi].add(mmsi)

        # Report how many new models were loaded from this bundle
        new_models_count = len(models) - previously_loaded_count
        print(f"Loaded {new_models_count} new models from {bundle_name}")

        # If we've loaded all models we need, we can stop
        if len(models) >= len(gp_regression_dataset):
            print(
                f"Reached target of {len(gp_regression_dataset)} models. Stopping bundle loading."
            )
            break

    print(f"Successfully loaded {len(models)} models from partial bundles")
    return models, likelihoods, losses


def train_remaining_models(models, likelihoods, losses, gp_regression_dataset, device):
    """Train models for any MMSIs not loaded from bundles."""
    # Find remaining MMSIs that need training
    remaining_mmsis = []
    for idx, entry in enumerate(gp_regression_dataset):
        mmsi = entry[0]
        if mmsi not in models:
            remaining_mmsis.append((idx, mmsi))

    if not remaining_mmsis:
        print(
            "All required models loaded from partial bundles. No additional training needed."
        )
        return models, likelihoods, losses

    print(f"Training {len(remaining_mmsis)} additional models for missing MMSIs...")

    # Track timing information for ETA calculation
    start_time = time.time()
    processed_count = 0
    avg_time_per_model = None

    # Create outer progress bar for overall tracking
    overall_pbar = tqdm(
        total=len(remaining_mmsis), desc="Training Additional Models", position=0
    )

    try:
        for idx, mmsi in remaining_mmsis:
            # Clear previous iteration output
            clear_output(wait=True)

            # Get the data for this MMSI
            entry = gp_regression_dataset[idx]
            _, times, state_trajectory = entry

            # Recalculate and display ETA
            if processed_count > 0:
                elapsed = time.time() - start_time
                avg_time_per_model = elapsed / processed_count
                remaining_models = len(remaining_mmsis) - processed_count
                eta_seconds = avg_time_per_model * remaining_models
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

                # Display progress summary
                print(
                    f"Training Additional Models: {processed_count}/{len(remaining_mmsis)} completed"
                )
                print(f"Average time per model: {avg_time_per_model:.2f} seconds")
                print(f"Estimated time remaining: {eta_str}")
            else:
                print(f"Training Additional Models: Starting...")

            # Show overall progress bar
            overall_pbar.display()

            print(
                f"\nFitting model for MMSI {mmsi} (#{processed_count+1}/{len(remaining_mmsis)})"
            )

            X = times.detach().unsqueeze(1).to(device)
            Y = state_trajectory.detach().to(device)

            num_outputs = Y.shape[1]

            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=num_outputs
            ).to(device)

            model = MultiOutputExactGPModel(
                X, Y, likelihood, num_outputs=num_outputs
            ).to(device)

            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            with gpytorch.settings.cholesky_jitter(1e-3):
                loss, model, likelihood = train_model(
                    model,
                    likelihood,
                    X,
                    Y,
                    num_epochs=50,
                    lr=0.1,
                    mmsi=mmsi,
                    session_id=session_id,
                )

            # Store results
            models[mmsi] = model
            likelihoods[mmsi] = likelihood
            losses[mmsi] = loss.item()

            # Update counters and progress
            processed_count += 1
            overall_pbar.update(1)

    finally:
        overall_pbar.close()

    # Display completion message with timing stats
    if processed_count > 0:
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Successfully trained {processed_count} additional models")
        print(
            f"Total time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        )
        print(f"Average time per model: {total_time/processed_count:.2f} seconds")

    print(f"Total models: {len(models)}")
    return models, likelihoods, losses


def train_all_models(gp_regression_dataset, device):
    """Train all models from scratch."""
    models = {}
    likelihoods = {}
    losses = {}

    print("Starting training process...")

    # Track timing information for ETA calculation
    start_time = time.time()
    processed_count = 0
    avg_time_per_model = None

    # Create outer progress bar for overall tracking
    overall_pbar = tqdm(
        total=len(gp_regression_dataset), desc="Overall Progress", position=0
    )

    try:
        for idx in range(len(gp_regression_dataset)):
            # Clear previous iteration output
            clear_output(wait=True)

            # Recalculate and display ETA
            if processed_count > 0:
                elapsed = time.time() - start_time
                avg_time_per_model = elapsed / processed_count
                remaining_models = len(gp_regression_dataset) - processed_count
                eta_seconds = avg_time_per_model * remaining_models
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

                # Display progress summary
                print(
                    f"Fitting GP Models: {processed_count}/{len(gp_regression_dataset)} completed"
                )
                print(f"Average time per model: {avg_time_per_model:.2f} seconds")
                print(f"Estimated time remaining: {eta_str}")
            else:
                print(f"Fitting GP Models: Starting...")

            # Show overall progress bar
            overall_pbar.display()

            # Get data for current model
            mmsi, times, state_trajectory = gp_regression_dataset[idx]
            print(
                f"\nFitting model for MMSI {mmsi} (#{idx+1}/{len(gp_regression_dataset)})"
            )

            X = times.detach().unsqueeze(1).to(device)
            Y = state_trajectory.detach().to(device)

            num_outputs = Y.shape[1]

            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=num_outputs
            ).to(device)

            model = MultiOutputExactGPModel(
                X, Y, likelihood, num_outputs=num_outputs
            ).to(device)

            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            with gpytorch.settings.cholesky_jitter(1e-3):
                loss, model, likelihood = train_model(
                    model,
                    likelihood,
                    X,
                    Y,
                    num_epochs=50,
                    lr=0.1,
                    mmsi=mmsi,
                    session_id=session_id,
                )

            # Store results
            models[mmsi] = model
            likelihoods[mmsi] = likelihood
            losses[mmsi] = loss.item()

            # Update counters and progress
            processed_count += 1
            overall_pbar.update(1)

    finally:
        overall_pbar.close()

    # Display completion message with timing stats
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    clear_output(wait=True)
    print(
        f"Model fitting complete! {len(models)}/{len(gp_regression_dataset)} models successfully fit"
    )
    print(f"Total time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"Average time per model: {total_time/processed_count:.2f} seconds")

    return models, likelihoods, losses


def load_or_train_models(
    date_range, gp_regression_dataset, device, reload_existing_models=True
):
    """Main function to load or train GP models for the given date range."""
    num_trajectories = len(gp_regression_dataset)
    models = {}
    likelihoods = {}
    losses = {}

    # Create a filename based on the dataset's date range and number of trajectories
    start_date = date_range[0].strftime("%Y-%m-%d")
    end_date = date_range[-1].strftime("%Y-%m-%d")

    # If it's a single day, use just one date, otherwise show the range
    if start_date == end_date:
        date_identifier = start_date
    else:
        date_identifier = f"{start_date}_to_{end_date}"

    # Define model file path with number of trajectories included
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/rbf_and_linear_kernel_classification_bundle_{date_identifier}_n{num_trajectories}.pt"

    # First check if models with the exact date range already exist
    if os.path.exists(save_path) and reload_existing_models:
        print(
            f"Found existing model bundle for date range {date_identifier} with {num_trajectories} trajectories"
        )
        print(f"Loading models from {save_path} instead of retraining...")
        models, likelihoods, losses = load_exact_model_bundle(
            save_path, gp_regression_dataset, device
        )

    elif reload_existing_models:
        print(
            f"No exact match found for date range {date_identifier}. Looking for partial date bundles..."
        )

        # Check for models that cover any days within our date range
        partial_bundles = find_partial_bundles(save_dir, date_range)

        if partial_bundles:
            # Load from partial bundles with prioritization and train any remaining models
            models, likelihoods, losses = load_prioritized_bundles(
                partial_bundles, gp_regression_dataset, device
            )
            models, likelihoods, losses = train_remaining_models(
                models, likelihoods, losses, gp_regression_dataset, device
            )

            # Save the complete bundle
            save_model_bundle(models, likelihoods, losses, date_range, save_path)
        else:
            print("No partial bundles found. Training all models from scratch...")
            models, likelihoods, losses = train_all_models(
                gp_regression_dataset, device
            )
            save_model_bundle(models, likelihoods, losses, date_range, save_path)
    else:
        # Check if we need to train new models
        if os.path.exists(save_path) and not reload_existing_models:
            print(
                f"Models exist at {save_path} but reload_existing_models=False, retraining..."
            )
        else:
            print(
                f"No existing models found for date range {date_identifier} with {num_trajectories} trajectories"
            )

        models, likelihoods, losses = train_all_models(gp_regression_dataset, device)
        save_model_bundle(models, likelihoods, losses, date_range, save_path)

    return models, likelihoods, losses
