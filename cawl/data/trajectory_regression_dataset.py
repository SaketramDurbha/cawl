import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm


class AISTrajectoryRegressionDataset(Dataset):
    """
    Dataset for trajectory classification of AIS data.
    """

    def __init__(
        self, date_range, device, scale_trajectories=True, mode: str = "classification"
    ):  # TODO: handle multiple csv imports
        self.KNOTS_TO_METERS_PER_SECOND = 0.514444
        self.device = device

        # Check if the date range df has already been processed and cached
        start_str = date_range[0].strftime("%Y_%m_%d")
        end_str = date_range[-1].strftime("%Y_%m_%d")
        processed_data_dir = "data/processed"
        cache_filename = os.path.join(
            processed_data_dir, f"processed_AIS_df_{start_str}_{end_str}.pkl"
        )

        if os.path.exists(cache_filename):
            print(f"Loading cached dataframe from {cache_filename}")
            with open(cache_filename, "rb") as f:
                cache = pickle.load(f)
                self.df = cache["df"]
                self.trajectories_by_mmsi = cache["trajectories_by_mmsi"]
        else:
            self.create_combined_df(date_range)  # combined csvs into self.df
            self.process_AIS_data()
            os.makedirs(processed_data_dir, exist_ok=True)

            with open(cache_filename, "wb") as f:
                pickle.dump(
                    {"df": self.df, "trajectories_by_mmsi": self.trajectories_by_mmsi},
                    f,
                )
            print(f"Saved processed dataset to: {cache_filename}")

        self.scale_trajectories()
        self.print_stats()

        # Initialize vessel group mappings
        self._init_vessel_group_mappings()

        # self.df = pd.read_csv(csv_path)
        # train_df, test_df = train_test_split(self.df, test_size=0.2)

    def create_combined_df(self, date_range):
        """
        Create a combined dataframe from multiple CSV files.
        """
        # Create a list to hold the dataframes
        df_list = []
        csv_folder_path = "data/"
        for date in tqdm(date_range, desc="Combining AIS CSVs into a single df"):
            csv_filename = f"AIS_{date.year}_{date.month:02d}_{date.day:02d}.csv"
            csv_path = os.path.join(csv_folder_path, csv_filename)
            if os.path.exists(csv_path):
                logging.info(f"Loading {csv_filename}")
                daily_df = pd.read_csv(csv_path)
                df_list.append(daily_df)
            else:
                logging.warning(
                    f"File {csv_filename} not found. Make sure to download the desired date ranges via cli_ais_dataloader.py as described in the README. Skipping."
                )

        if df_list:
            self.df = pd.concat(df_list, ignore_index=True)
        else:
            raise ValueError("No CSV files found in the specified date range.")

    def _init_vessel_group_mappings(self):
        """
        Maps AIS vessel type codes to human-readable group names and IDs.
        """
        vessel_groups = {}
        vessel_groups["Cargo"] = np.concatenate([np.arange(70, 80), [1003, 1004, 1016]])
        vessel_groups["Fishing"] = np.array([30, 1001, 1002])
        vessel_groups["Military"] = np.array([35])
        vessel_groups["Not Available"] = np.array([0])
        vessel_groups["Other"] = np.concatenate(
            [
                np.arange(1, 21),
                np.arange(23, 30),
                np.arange(33, 35),
                np.arange(38, 52),
                np.arange(53, 60),
                np.arange(90, 1000),
                np.arange(1005, 1012),
                [1018, 1020, 1022],
            ]
        )
        vessel_groups["Passenger"] = np.concatenate(
            [np.arange(60, 70), np.arange(1012, 1016)]
        )
        vessel_groups["Pleasure Craft"] = np.array([36, 37, 1019])
        vessel_groups["Tanker"] = np.concatenate([np.arange(80, 90), [1017, 1024]])
        vessel_groups["Tug Tow"] = np.array([21, 22, 31, 32, 52, 1023, 1025])

        # Map type code to group name
        self.map_to_vessel_group = {}
        for vg in vessel_groups:
            for i in vessel_groups[vg]:
                self.map_to_vessel_group[int(i)] = vg

        # Map group name to group ID
        self.vessel_group_to_id = {
            v: i for i, v in enumerate(sorted(set(self.map_to_vessel_group.values())))
        }

        # Map group ID to group name
        self.vessel_group_id_to_group = {
            i: v for v, i in self.vessel_group_to_id.items()
        }

    def get_vessel_group_by_mmsi(self, mmsi):
        """
        Given an MMSI, return the vessel group name.
        """
        # Find the first occurrence of this MMSI in the dataframe
        row = self.df[self.df["MMSI"] == mmsi]  # Filter by MMSI
        if row.empty:
            return "Unknown"
        type_code = int(row.iloc[0]["VesselType"])
        return self.map_to_vessel_group.get(type_code, "Unknown")

    def get_vessel_group_id_by_mmsi(self, mmsi):
        """
        Given an MMSI, return the vessel group ID.
        """
        group_name = self.get_vessel_group_by_mmsi(mmsi)
        return self.vessel_group_to_id.get(group_name, -1)

    def process_AIS_data(self):
        print("Processing AIS data...")
        self.remove_AIS_artifacts()
        self.df["BaseDateTime"] = pd.to_datetime(self.df["BaseDateTime"])

        # Group by MMSI and compute seconds since start for each group
        self.df["SecondsSinceStart"] = 0  # Initialize column
        self.trajectories_by_mmsi = []
        self.MMSI_groups = self.df.groupby("MMSI")

        for mmsi, group in tqdm(
            self.MMSI_groups, desc="Standardizing trajectories for MMSI groups"
        ):
            group = group.sort_values(by="BaseDateTime")  # Ensure sorted by time
            earliest_time = group["BaseDateTime"].min()
            group["SecondsSinceStart"] = (
                group["BaseDateTime"] - earliest_time
            ).dt.total_seconds()
            self.df.loc[group.index, "SecondsSinceStart"] = group["SecondsSinceStart"]

            seconds_since_start = group["SecondsSinceStart"].values

            state_space_trajectory = np.array(
                [
                    self.ais_to_state_space(
                        row["LON"], row["LAT"], row["Heading"], row["SOG"]
                    )
                    for _, row in group.iterrows()
                ]
            )

            # Compute phi_dot for the group
            heading = group["Heading"].values
            times = group["SecondsSinceStart"].values
            phi_dot = np.zeros(len(heading))
            if len(heading) > 1:
                dt = np.diff(times)
                d_heading = np.diff(heading)
                # Avoid division by zero
                dt[dt == 0] = np.nan
                phi_dot[1:] = d_heading / dt
                # Replace NaNs and infs with 0
                phi_dot = np.nan_to_num(phi_dot, nan=0.0, posinf=0.0, neginf=0.0)
                phi_dot[0] = phi_dot[1] if len(phi_dot) > 1 else 0
            else:
                phi_dot[0] = 0  # Default to 0 if there's only one entry

            # Store the state space trajectories for every MMSI
            self.trajectories_by_mmsi.append(
                (mmsi, seconds_since_start, state_space_trajectory)
            )

    def ais_to_state_space(self, LON, LAT, Heading, SOG):
        x = 111320 * LON
        y = 111320 * LAT
        theta = np.radians(Heading)

        x_dot = 0.514444 * SOG * np.cos(theta)  # Convert to m/s
        y_dot = 0.514444 * SOG * np.sin(theta)  # Convert to m/s

        # TODO: add angular rate phid

        # Calculate phi_dot (angular rate) using finite differencing
        # if len(self.df) > 1:
        #     phi_dot = np.zeros(len(self.df))
        #     phi_dot[1:] = np.diff(self.df['Heading']) / np.diff(self.df['SecondsSinceStart'])
        #     phi_dot[0] = phi_dot[1]  # Reuse the second value for the first entry
        # else:
        #     phi_dot = np.array([0])  # Default to 0 if there's only one entry

        # NOTE: phi_dot cannot be computed per element since it is not in the AIS data, so it is computed via finite diff in process_AIS_data
        return np.array([x, y, theta, x_dot, y_dot, 0])

    def remove_AIS_artifacts(self):
        """
        Remove artifacts from the AIS data. Removed based on invalid msgs as defined by: https://www.navcen.uscg.gov/ais-class-a-reports
        """
        start_row_count = self.df.shape[0]
        self.df = self.df[self.df["COG"] != 360]
        self.df = self.df[self.df["SOG"] != 102.3]
        self.df = self.df[self.df["LAT"] != 181]
        self.df = self.df[self.df["Heading"] != 511]
        self.df.dropna(inplace=True)
        end_row_count = self.df.shape[0]

        print(
            f"Removed {start_row_count - end_row_count} out of {start_row_count} rows due to invalid COG, SOG, LAT, or Heading values."
        )

    def scale_trajectories(self):
        """
        Scale each MMSI's trajectory independently and store the scalers.
        """
        self.scalers_by_mmsi = {}
        scaled_trajectories = []

        for mmsi, times, traj in tqdm(
            self.trajectories_by_mmsi, desc="Scaling trajectories for each MMSI"
        ):
            # Fit scalers for this ship
            state_scaler = StandardScaler().fit(traj)
            time_scaler = StandardScaler().fit(times.reshape(-1, 1))
            # Transform
            scaled_traj = state_scaler.transform(traj)
            scaled_times = time_scaler.transform(times.reshape(-1, 1)).flatten()
            # Store
            scaled_trajectories.append((mmsi, scaled_times, scaled_traj))
            self.scalers_by_mmsi[mmsi] = {
                "state_scaler": state_scaler,
                "time_scaler": time_scaler,
            }

        self.scaled_trajectories_by_mmsi = scaled_trajectories

    # def inverse_transform(self, mmsi, scaled_times, scaled_traj):
    #     """
    #     Inverse transform scaled data for a given MMSI.
    #     """
    #     state_scaler = self.scalers_by_mmsi[mmsi]['state_scaler']
    #     time_scaler = self.scalers_by_mmsi[mmsi]['time_scaler']
    #     orig_traj = state_scaler.inverse_transform(scaled_traj)
    #     orig_times = time_scaler.inverse_transform(scaled_times.reshape(-1, 1)).flatten()
    #     return orig_times, orig_traj
    def print_stats(self):
        """
        Print statistics about the dataset.
        """
        print("\n===== Dataset Statistics =====")
        print(f"Total number of AIS messages: {len(self.df)}")
        print(f"Number of unique MMSIs: {len(self.trajectories_by_mmsi)}")
        print(
            f"Date range: {self.df['BaseDateTime'].min()} to {self.df['BaseDateTime'].max()}"
        )

    def plot_vessel_group_histogram(
        self, figsize=(12, 6), show_counts=True, return_counts=False, fontsize=16
    ):
        """
        Plot a histogram of vessel groups in the dataset with increased font size.

        Args:
            figsize: Tuple specifying figure dimensions
            show_counts: Whether to display count values above bars
            return_counts: If True, return the count data
            fontsize: Base font size for plot elements (default=14)

        Returns:
            If return_counts is True, returns a pandas Series with group counts
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        # Set a clean style with larger font sizes
        sns.set_theme(style="whitegrid", font_scale=1.2)

        # Get all unique MMSIs
        mmsi_values = pd.unique(self.df["MMSI"].values)
        vessel_groups = []

        # Get group for each MMSI
        for mmsi in mmsi_values:
            group = self.get_vessel_group_by_mmsi(mmsi)
            vessel_groups.append((mmsi, group))

        # Convert to DataFrame for analysis
        groups_df = pd.DataFrame(vessel_groups, columns=["MMSI", "Group"])
        group_counts = groups_df["Group"].value_counts().sort_index()

        # Create the plot with increased font size
        plt.figure(figsize=figsize)
        ax = group_counts.plot(kind="bar", color="tab:blue")

        # Increase font sizes
        plt.title(
            "Distribution of Vessel Types", fontsize=fontsize + 4, fontweight="bold"
        )
        plt.xlabel("Vessel Group", fontsize=fontsize + 2)
        plt.ylabel("Count", fontsize=fontsize + 2)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Increase tick label font size
        plt.xticks(rotation=45, ha="right", fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # Add count labels if requested
        if show_counts:
            for i, count in enumerate(group_counts):
                plt.text(i, count + 0.5, str(count), ha="center", fontsize=fontsize - 1)

        # Adjust layout to make room for the larger fonts
        plt.tight_layout()

        # Print summary statistics
        print(f"Number of vessel groups: {len(group_counts)}")

        if return_counts:
            return group_counts

    def get_all_mmsis(self):
        """Return a list of all unique MMSI values in the dataset."""
        return pd.unique(self.df["MMSI"].values).tolist()

    def __len__(self):
        return len(
            self.trajectories_by_mmsi
        )  # subtract 1 since this is single step state prediction

    def __getitem__(self, idx):
        if self.scale_trajectories:
            mmsi, times, state_trajectory = self.scaled_trajectories_by_mmsi[idx]
        else:
            mmsi, times, state_trajectory = self.trajectories_by_mmsi[idx]

        return (
            mmsi,
            torch.from_numpy(times).float().to(self.device),
            torch.from_numpy(state_trajectory).float().to(self.device),
        )

    # def __getitem__(self, idx):
    #     dt = self.df.iloc[idx+1] - [idx]

    #     # Get state at timestep k
    #     x_k = self.df.iloc[idx]['LON']
    #     y_k = self.df.iloc[idx]['LAT']
    #     phi_k = self.df.iloc[idx]['HEADING']
    #     COG = self.df.iloc[idx]['COG'] # angle that the instantaneous velocity makes in world coords
    #     SOG = self.df.iloc[idx]['SOG'] # speed of a vessel in relation to a fixed point on the Earth's surface
    #     xd_k = np.cos(COG) * SOG * self.KNOTS_TO_METERS_PER_SECOND
    #     yd_k = np.sin(COG) * SOG * self.KNOTS_TO_METERS_PER_SECOND
    #     phid_k = 0 # TODO: how to better estimate angular rate?

    #     # Get state at timestep kp1
    #     x_kp1 = self.df.iloc[idx]['LON']
    #     y_kp1 = self.df.iloc[idx]['LAT']
    #     phi_kp1 = self.df.iloc[idx]['HEADING']
    #     COG_kp1 = self.df.iloc[idx]['COG'] # angle that the instantaneous velocity makes in world coords
    #     SOG_kp1 = self.df.iloc[idx]['SOG'] # speed of a vessel in relation to a fixed point on the Earth's surface
    #     xd_kp1 = np.cos(COG_kp1) * SOG_kp1 * self.KNOTS_TO_METERS_PER_SECOND
    #     yd_kp1 = np.sin(COG_kp1) * SOG_kp1 * self.KNOTS_TO_METERS_PER_SECOND
    #     phid_kp1 = 0 # TODO: how to better estimate angular rate?

    #     # x = [x, y, phi, xd, yd, phid]
    #     state_k = np.array([x_k, y_k, phi_k, xd_k, yd_k, phid_k])
    #     state_kp1 = np.array(x_kp1, y_kp1, phi_kp1, xd_kp1, yd_kp1, phid_kp1)

    # return state_k, state_kp1
