from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from cawl.data.gp_kernel_dataset import extract_data


class PerClassDecisionTreeModel:
    """
    Per-class decision tree model that trains separate trees for each class.

    This model implements the following approach:
    1. For each class, train multiple decision trees using one-vs-rest classification
    2. Each tree predicts whether a sample belongs to its target class or not
    3. Use confidence thresholds to abstain on uncertain predictions
    4. Generate weak labels only when trees are confident about their predictions
    """

    def __init__(
        self,
        dataset,
        max_depth: int = 3,
        n_trees_per_class: int = 5,
        feature_fraction: float = 0.5,
        confidence_threshold: float = 0.8,
        random_seed: int = 42,
    ):
        """
        Initialize the per-class decision tree model.

        Args:
            dataset: GPKernelShipClassificationDataset instance
            max_depth: Maximum depth of each decision tree
            n_trees_per_class: Number of trees to train for each class
            feature_fraction: Fraction of features to use for each tree (0.0 to 1.0)
            confidence_threshold: Minimum confidence required to make a prediction (0.0 to 1.0)
            random_seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.max_depth = max_depth
        self.n_trees_per_class = n_trees_per_class
        self.feature_fraction = feature_fraction
        self.confidence_threshold = confidence_threshold
        self.random_state = random_seed

        # Extract features and labels from dataset
        self.features, self.labels, self.mmsis = extract_data(dataset)
        self.n_features = self.features.shape[1]
        self.classes = np.unique(self.labels)
        self.n_classes = len(self.classes)

        # Store trained trees per class
        self.trees_per_class: Dict[int, List[DecisionTreeClassifier]] = {}
        self.feature_indices_per_tree: Dict[int, List[np.ndarray]] = {}
        self.sample_indices_per_tree: Dict[int, List[np.ndarray]] = {}

        print(
            f"Dataset loaded: {len(self.features)} samples, "
            f"{self.n_features} features, {self.n_classes} classes"
        )
        print(
            f"Training {self.n_trees_per_class} trees per class = {self.n_trees_per_class * self.n_classes} total trees"
        )
        print(f"Confidence threshold: {self.confidence_threshold}")

    def _resample_with_replacement(
        self, n_samples: int, target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Resample n_samples from the dataset with replacement.

        Args:
            n_samples: Number of samples to resample
            target_class: If provided, ensures balanced sampling for this class
        """
        if target_class is not None:
            # Ensure balanced sampling for the target class
            target_class_indices = np.where(self.labels == target_class)[0]
            other_class_indices = np.where(self.labels != target_class)[0]

            if len(target_class_indices) == 0:
                raise ValueError(
                    f"No samples of class {target_class} exist in the dataset"
                )
            if len(other_class_indices) == 0:
                raise ValueError(
                    f"No samples of classes other than {target_class} exist in the dataset"
                )

            # Calculate how many samples to take from each class
            # Ensure at least 20% from each class to maintain balance
            min_target_samples = max(1, int(n_samples * 0.2))
            min_other_samples = max(1, int(n_samples * 0.2))

            # Sample from target class
            target_samples = np.random.choice(
                target_class_indices,
                size=min(min_target_samples, len(target_class_indices)),
                replace=True,
            )

            # Sample from other classes
            other_samples = np.random.choice(
                other_class_indices,
                size=min(min_other_samples, len(other_class_indices)),
                replace=True,
            )

            # Combine and shuffle
            balanced_indices = np.concatenate([target_samples, other_samples])
            np.random.shuffle(balanced_indices)

            # If we need more samples, fill with random sampling
            if len(balanced_indices) < n_samples:
                remaining_samples = n_samples - len(balanced_indices)
                additional_samples = np.random.choice(
                    len(self.features), size=remaining_samples, replace=True
                )
                balanced_indices = np.concatenate(
                    [balanced_indices, additional_samples]
                )

            return balanced_indices[:n_samples]
        else:
            # Standard resampling
            return np.random.choice(len(self.features), size=n_samples, replace=True)

    def _select_random_features(self) -> np.ndarray:
        """Select a random fraction of features."""
        n_features_to_select = max(1, int(self.n_features * self.feature_fraction))
        return np.random.choice(
            self.n_features, size=n_features_to_select, replace=False
        )

    def _create_one_vs_rest_labels(self, target_class: int) -> np.ndarray:
        """Create binary labels for one-vs-rest classification."""
        binary_labels = np.zeros(len(self.labels), dtype=int)
        binary_labels[self.labels == target_class] = 1
        return binary_labels

    def train_single_tree(
        self, sample_indices: np.ndarray, feature_indices: np.ndarray, target_class: int
    ) -> DecisionTreeClassifier:
        """Train a single decision tree for one-vs-rest classification."""
        # Extract resampled data
        X_resampled = self.features[sample_indices][:, feature_indices]
        y_resampled = self._create_one_vs_rest_labels(target_class)[sample_indices]

        # Check if we have samples from both classes
        unique_classes = np.unique(y_resampled)
        if len(unique_classes) < 2:
            # If we don't have both classes, we need to ensure we have at least one sample from each
            # This can happen when resampling doesn't include any samples from the target class
            # Find indices of target class samples in the original dataset
            target_class_indices = np.where(self.labels == target_class)[0]
            other_class_indices = np.where(self.labels != target_class)[0]

            # Ensure we have at least one sample from each class
            min_samples_per_class = 1
            target_samples_needed = min(
                min_samples_per_class, len(target_class_indices)
            )
            other_samples_needed = min(min_samples_per_class, len(other_class_indices))

            # Sample from target class
            target_samples = np.random.choice(
                target_class_indices, size=target_samples_needed, replace=True
            )
            # Sample from other classes
            other_samples = np.random.choice(
                other_class_indices, size=other_samples_needed, replace=True
            )

            # Combine samples
            balanced_indices = np.concatenate([target_samples, other_samples])

            # Update the resampled data
            X_resampled = self.features[balanced_indices][:, feature_indices]
            y_resampled = self._create_one_vs_rest_labels(target_class)[
                balanced_indices
            ]

        # Train decision tree
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth, random_state=self.random_state, criterion="gini"
        )
        tree.fit(X_resampled, y_resampled)

        return tree

    def train_all_trees(self, tree_train_indices: Optional[np.ndarray] = None) -> None:
        """
        Train multiple decision trees for each class using one-vs-rest approach.

        Args:
            tree_train_indices: Optional array of indices to use for training.
                              If None, uses all available data.
        """
        if tree_train_indices is None:
            tree_train_indices = np.arange(len(self.features))

        n_train_samples = len(tree_train_indices)
        print(
            f"Training {self.n_trees_per_class} trees for each of {self.n_classes} classes..."
        )
        print(f"Training set size: {n_train_samples}")
        print(f"Feature fraction per tree: {self.feature_fraction}")
        print(f"Max tree depth: {self.max_depth}")

        # Initialize storage for each class
        for class_id in self.classes:
            self.trees_per_class[class_id] = []
            self.feature_indices_per_tree[class_id] = []
            self.sample_indices_per_tree[class_id] = []

        # Train trees for each class
        for class_id in tqdm(self.classes, desc="Training trees per class"):
            for i in range(self.n_trees_per_class):
                # Resample with replacement
                resampled_indices = self._resample_with_replacement(
                    n_train_samples, target_class=class_id
                )
                # Map back to original indices
                sample_indices = tree_train_indices[resampled_indices]

                # Select random features
                feature_indices = self._select_random_features()

                # Train tree for this class
                tree = self.train_single_tree(sample_indices, feature_indices, class_id)

                # Store results
                self.trees_per_class[class_id].append(tree)
                self.feature_indices_per_tree[class_id].append(feature_indices)
                self.sample_indices_per_tree[class_id].append(sample_indices)

        total_trees = sum(len(trees) for trees in self.trees_per_class.values())
        print(
            f"Successfully trained {total_trees} trees ({self.n_trees_per_class} per class)"
        )

    def predict_single_tree(
        self, X: np.ndarray, class_id: int, tree_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction using a single tree for a specific class.

        Returns:
            Tuple of (predictions, confidences)
            - predictions: 1 if predicted as target class, 0 otherwise, -1 if abstained
            - confidences: confidence scores for the predictions
        """
        tree = self.trees_per_class[class_id][tree_idx]
        feature_indices = self.feature_indices_per_tree[class_id][tree_idx]
        X_subset = X[:, feature_indices]

        # Get probabilities
        proba = tree.predict_proba(X_subset)

        # For binary classification, proba[:, 1] is the probability of the positive class
        confidences = proba[:, 1]  # Probability of belonging to target class

        # Make predictions based on confidence threshold
        predictions = np.zeros(X.shape[0], dtype=int)
        predictions[confidences >= self.confidence_threshold] = 1  # Target class
        predictions[confidences < (1 - self.confidence_threshold)] = (
            0  # Not target class
        )
        predictions[
            (confidences >= (1 - self.confidence_threshold))
            & (confidences < self.confidence_threshold)
        ] = -1  # Abstain

        return predictions, confidences

    def predict_ensemble(self, X: np.ndarray, method: str | None = None) -> np.ndarray:
        """
        Make ensemble predictions using all trees for all classes.

        Args:
            X: Input features
            method: 'majority' for majority voting, 'confidence' for confidence-weighted voting

        Returns:
            Predicted labels (-1 for abstained predictions)
        """
        if method == "majority" or method is None:
            return self._predict_majority_voting(X, method is None)
        elif method == "confidence":
            return self._predict_confidence_weighted(X)
        else:
            raise ValueError("Method must be None or 'majority' or 'confidence'")

    def _predict_majority_voting(
        self, X: np.ndarray, none_method: bool = False
    ) -> np.ndarray:
        """Predict using majority voting across all trees."""
        # Collect predictions from all trees for all classes
        all_predictions = []

        for class_id in self.classes:
            for tree_idx in range(len(self.trees_per_class[class_id])):
                predictions, _ = self.predict_single_tree(X, class_id, tree_idx)
                # Convert to class labels: 1 -> class_id, 0 -> -1 (not this class), -1 -> -1 (abstain)
                class_predictions = np.full(X.shape[0], -1, dtype=int)
                class_predictions[predictions == 1] = class_id
                all_predictions.append(class_predictions)

        all_predictions = np.array(all_predictions)
        if none_method:
            return all_predictions.transpose()

        # Majority voting for each sample
        final_predictions = []
        for j in range(X.shape[0]):
            votes = all_predictions[:, j]
            # Remove abstentions (-1) from voting
            valid_votes = votes[votes != -1]

            if len(valid_votes) == 0:
                final_predictions.append(-1)  # All trees abstained
            else:
                # Get most common prediction
                unique, counts = np.unique(valid_votes, return_counts=True)
                final_predictions.append(unique[np.argmax(counts)])

        return np.array(final_predictions)

    def _predict_confidence_weighted(self, X: np.ndarray) -> np.ndarray:
        """Predict using confidence-weighted voting."""
        # Collect predictions and confidences from all trees
        class_scores = {class_id: np.zeros(X.shape[0]) for class_id in self.classes}
        total_confidence = np.zeros(X.shape[0])

        for class_id in self.classes:
            for tree_idx in range(len(self.trees_per_class[class_id])):
                predictions, confidences = self.predict_single_tree(
                    X, class_id, tree_idx
                )

                # Add confidence scores for positive predictions
                positive_mask = predictions == 1
                class_scores[class_id][positive_mask] += confidences[positive_mask]
                total_confidence[positive_mask] += confidences[positive_mask]

        # Make final predictions
        final_predictions = np.full(X.shape[0], -1, dtype=int)

        for i in range(X.shape[0]):
            if total_confidence[i] > 0:
                # Find class with highest score
                best_class = max(class_scores.keys(), key=lambda c: class_scores[c][i])
                final_predictions[i] = best_class

        return final_predictions

    def evaluate_ensemble(
        self, X_test: np.ndarray, y_test: np.ndarray, method: str = "majority"
    ) -> Dict:
        """
        Evaluate the ensemble on test data.

        Args:
            X_test: Test features
            y_test: Test labels
            method: Prediction method ('majority' or 'confidence')

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict_ensemble(X_test, method=method)

        # Calculate accuracy excluding abstentions
        non_abstained_mask = y_pred != -1
        if np.sum(non_abstained_mask) > 0:
            accuracy = accuracy_score(
                y_test[non_abstained_mask], y_pred[non_abstained_mask]
            )
        else:
            accuracy = 0.0

        # Calculate coverage (percentage of non-abstained predictions)
        coverage = np.mean(non_abstained_mask)

        results = {
            "accuracy": accuracy,
            "coverage": coverage,
            "predictions": y_pred,
            "true_labels": y_test,
            "method": method,
            "n_abstained": np.sum(~non_abstained_mask),
            "n_predicted": np.sum(non_abstained_mask),
        }

        return results

    def get_tree_statistics(self) -> Dict:
        """Get statistics about the trained trees."""
        all_depths = []
        all_leaves = []
        all_feature_indices = []

        for class_id in self.classes:
            for tree in self.trees_per_class[class_id]:
                all_depths.append(tree.get_depth())
                all_leaves.append(tree.get_n_leaves())

            for indices in self.feature_indices_per_tree[class_id]:
                all_feature_indices.extend(indices)

        # Feature usage statistics
        feature_counts = {}
        for idx in all_feature_indices:
            feature_counts[idx] = feature_counts.get(idx, 0) + 1

        return {
            "n_classes": self.n_classes,
            "n_trees_per_class": self.n_trees_per_class,
            "total_trees": len(all_depths),
            "avg_depth": np.mean(all_depths),
            "std_depth": np.std(all_depths),
            "avg_leaves": np.mean(all_leaves),
            "std_leaves": np.std(all_leaves),
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
        plt.title(f"Feature Usage in {stats['total_trees']} Per-Class Decision Trees")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def plot_class_coverage(self, X_test: np.ndarray, y_test: np.ndarray):
        """Plot coverage statistics per class."""
        class_coverage = {}
        class_accuracy = {}

        for class_id in self.classes:
            # Get predictions for this class only
            class_predictions = []
            for tree_idx in range(len(self.trees_per_class[class_id])):
                predictions, _ = self.predict_single_tree(X_test, class_id, tree_idx)
                class_predictions.append(predictions)

            class_predictions = np.array(class_predictions)

            # Calculate coverage (percentage of non-abstained predictions)
            non_abstained = class_predictions != -1
            coverage = np.mean(non_abstained)
            class_coverage[class_id] = coverage

            # Calculate accuracy for non-abstained predictions
            if np.sum(non_abstained) > 0:
                # For one-vs-rest, accuracy is percentage of correct positive predictions
                correct_predictions = (class_predictions == 1) & (y_test == class_id)
                accuracy = (
                    np.sum(correct_predictions) / np.sum(class_predictions == 1)
                    if np.sum(class_predictions == 1) > 0
                    else 0
                )
                class_accuracy[class_id] = accuracy
            else:
                class_accuracy[class_id] = 0.0

        # Plot coverage and accuracy per class
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        classes = list(class_coverage.keys())
        coverages = list(class_coverage.values())
        accuracies = list(class_accuracy.values())

        ax1.bar(classes, coverages)
        ax1.set_title("Coverage per Class")
        ax1.set_ylabel("Coverage")
        ax1.set_xlabel("Class ID")

        ax2.bar(classes, accuracies)
        ax2.set_title("Accuracy per Class")
        ax2.set_ylabel("Accuracy")
        ax2.set_xlabel("Class ID")

        plt.tight_layout()
        plt.show()

        return class_coverage, class_accuracy
