from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from cawl.data.gp_kernel_dataset import extract_data


class SmallDecisionTreesModel:
    """
    Trainer for multiple small decision trees using the GPKernelShipClassificationDataset.

    This class implements the following approach:
    1. Resample tree-train set samples from the tree-train set with replacement
    2. Take a random fraction of the features
    3. Train a small decision tree (e.g., depth 2 or 3)
    4. Repeat to get multiple trees
    """

    def __init__(
        self,
        dataset,
        max_depth: int = 3,
        n_trees: int = 100,
        feature_fraction: float = 0.5,
        random_seed: int = 42,
    ):
        """
        Initialize the trainer.

        Args:
            dataset: GPKernelShipClassificationDataset instance
            max_depth: Maximum depth of each decision tree
            n_trees: Number of small trees to train
            feature_fraction: Fraction of features to use for each tree (0.0 to 1.0)
            random_state: Random seed for reproducibility
        """
        self.dataset = dataset
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.feature_fraction = feature_fraction
        self.random_state = random_seed
        # Extract features and labels from dataset
        self.features, self.labels, self.mmsis = extract_data(dataset)
        self.n_features = self.features.shape[1]
        self.n_classes = len(np.unique(self.labels))

        # Store trained trees
        self.trees: List[DecisionTreeClassifier] = []
        self.feature_indices_per_tree: List[np.ndarray] = []
        self.sample_indices_per_tree: List[np.ndarray] = []

        print(
            f"Dataset loaded: {len(self.features)} samples, "
            f"{self.n_features} features, {self.n_classes} classes"
        )

    def _resample_with_replacement(self, n_samples: int) -> np.ndarray:
        """Resample n_samples from the dataset with replacement."""
        return np.random.choice(len(self.features), size=n_samples, replace=True)

    def _select_random_features(self) -> np.ndarray:
        """Select a random fraction of features."""
        n_features_to_select = max(1, int(self.n_features * self.feature_fraction))
        return np.random.choice(
            self.n_features, size=n_features_to_select, replace=False
        )

    def train_single_tree(
        self, sample_indices: np.ndarray, feature_indices: np.ndarray
    ) -> DecisionTreeClassifier:
        """Train a single small decision tree."""
        # Extract resampled data
        X_resampled = self.features[sample_indices][:, feature_indices]
        y_resampled = self.labels[sample_indices]

        # Train decision tree
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth, random_state=self.random_state, criterion="gini"
        )
        tree.fit(X_resampled, y_resampled)

        return tree

    def train_all_trees(self, tree_train_indices: Optional[np.ndarray] = None) -> None:
        """
        Train multiple small decision trees.

        Args:
            tree_train_indices: Optional array of indices to use for training.
                              If None, uses all available data.
        """
        if tree_train_indices is None:
            tree_train_indices = np.arange(len(self.features))

        n_train_samples = len(tree_train_indices)
        print(f"Training {self.n_trees} small decision trees...")
        print(f"Training set size: {n_train_samples}")
        print(f"Feature fraction per tree: {self.feature_fraction}")
        print(f"Max tree depth: {self.max_depth}")

        self.trees = []
        self.feature_indices_per_tree = []
        self.sample_indices_per_tree = []

        for i in tqdm(range(self.n_trees), desc="Training trees"):
            # Resample with replacement
            resampled_indices = self._resample_with_replacement(n_train_samples)
            # Map back to original indices
            sample_indices = tree_train_indices[resampled_indices]

            # Select random features
            feature_indices = self._select_random_features()

            # Train tree
            tree = self.train_single_tree(sample_indices, feature_indices)

            # Store results
            self.trees.append(tree)
            self.feature_indices_per_tree.append(feature_indices)
            self.sample_indices_per_tree.append(sample_indices)

        print(f"Successfully trained {len(self.trees)} trees")

    def predict_single_tree(
        self, X: np.ndarray, tree_idx: int, threshold: int | None = None
    ) -> np.ndarray:
        """Make prediction using a single tree."""
        tree = self.trees[tree_idx]
        feature_indices = self.feature_indices_per_tree[tree_idx]
        X_subset = X[:, feature_indices]

        if threshold is None:
            return tree.predict(X_subset)

        proba = tree.predict_proba(X_subset)
        predict = proba.argmax(axis=1)
        predict[proba.max(axis=1) < threshold] = -1
        return predict

    def predict_ensemble(
        self, X: np.ndarray, method: str | None = None, threshold: int | None = None
    ) -> np.ndarray:
        """
        Make ensemble predictions using all trees.

        Args:
            X: Input features
            method: 'majority' for majority voting, 'probability' for probability averaging, None for no voting

        Returns:
            Predicted labels
        """
        if method == "majority" or method is None:
            predictions = []
            for i in range(len(self.trees)):
                pred = self.predict_single_tree(X, i, threshold)
                predictions.append(pred)

            # Majority voting
            predictions = np.array(predictions)
            if method is None:
                return np.transpose(predictions)

            ensemble_pred = []
            for j in range(X.shape[0]):
                votes = predictions[:, j]
                # Get most common prediction
                unique, counts = np.unique(votes, return_counts=True)
                ensemble_pred.append(unique[np.argmax(counts)])

            return np.array(ensemble_pred)

        elif method == "probability":
            # Average probabilities across trees
            all_probs = []
            for i in range(len(self.trees)):
                tree = self.trees[i]
                feature_indices = self.feature_indices_per_tree[i]
                X_subset = X[:, feature_indices]
                probs = tree.predict_proba(X_subset)
                all_probs.append(probs)

            avg_probs = np.mean(all_probs, axis=0)
            return np.argmax(avg_probs, axis=1)

        else:
            raise ValueError("Method must be 'majority' or 'probability'")

    def evaluate_ensemble(
        self, X_test: np.ndarray, y_test: np.ndarray, method: str = "majority"
    ) -> Dict:
        """
        Evaluate the ensemble on test data.

        Args:
            X_test: Test features
            y_test: Test labels
            method: Prediction method ('majority' or 'probability')

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict_ensemble(X_test, method=method)

        accuracy = accuracy_score(y_test, y_pred)

        results = {
            "accuracy": accuracy,
            "predictions": y_pred,
            "true_labels": y_test,
            "method": method,
        }

        return results

    def get_tree_statistics(self) -> Dict:
        """Get statistics about the trained trees."""
        depths = [tree.get_depth() for tree in self.trees]
        n_leaves = [tree.get_n_leaves() for tree in self.trees]

        # Feature usage statistics
        all_feature_indices = []
        for indices in self.feature_indices_per_tree:
            all_feature_indices.extend(indices)

        feature_counts = {}
        for idx in all_feature_indices:
            feature_counts[idx] = feature_counts.get(idx, 0) + 1

        return {
            "n_trees": len(self.trees),
            "avg_depth": np.mean(depths),
            "std_depth": np.std(depths),
            "avg_leaves": np.mean(n_leaves),
            "std_leaves": np.std(n_leaves),
            "feature_usage": feature_counts,
            "most_used_features": sorted(
                feature_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }

    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance based on usage frequency."""
        stats = self.get_tree_statistics()
        feature_usage = stats["feature_usage"]

        # Get parameter names
        param_names = self.dataset.get_parameter_names()

        # Sort features by usage
        sorted_features = sorted(
            feature_usage.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        features, counts = zip(*sorted_features)
        feature_names = [param_names[f] for f in features]

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(feature_names)), counts)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel("Number of trees using this feature")
        plt.title(f"Feature Usage in {len(self.trees)} Small Decision Trees")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def plot_accuracy_vs_trees(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        step: int = 5,
        method: str = "majority",
    ):
        """Plot how accuracy changes as more trees are added to the ensemble."""
        accuracies = []
        n_trees_list = list(range(1, len(self.trees) + 1, step))

        for n_trees in tqdm(n_trees_list, desc="Evaluating ensemble sizes"):
            # Temporarily store only the first n_trees
            original_trees = self.trees.copy()
            original_feature_indices = self.feature_indices_per_tree.copy()

            self.trees = self.trees[:n_trees]
            self.feature_indices_per_tree = self.feature_indices_per_tree[:n_trees]

            # Evaluate
            results = self.evaluate_ensemble(X_test, y_test, method=method)
            accuracies.append(results["accuracy"])

            # Restore original trees
            self.trees = original_trees
            self.feature_indices_per_tree = original_feature_indices

        plt.figure(figsize=(10, 6))
        plt.plot(n_trees_list, accuracies, "bo-", linewidth=2, markersize=6)
        plt.xlabel("Number of Trees")
        plt.ylabel("Accuracy")
        plt.title(f"Ensemble Accuracy vs Number of Trees ({method} voting)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return n_trees_list, accuracies
