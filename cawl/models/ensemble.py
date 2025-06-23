from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from cawl.models.tree.per_class_decision_tree_model import PerClassDecisionTreeModel
from cawl.models.wl.wl import MLP, train


class EnsembleModel:
    """
    Ensemble model that combines decision tree predictions with weak supervised classifier predictions.

    This model implements the following voting mechanism:
    votes[l] = 0.5 * soft-max-prob(weak-supervised-classifier, x) + 0.5 * #(trees that vote for l on input x)/#(trees that do not abstain on x)
    prediction for input x: argmax(votes)

    The model combines:
    1. Decision tree ensemble predictions (with abstention capability)
    2. Neural network predictions trained using weak supervision
    """

    def __init__(
        self,
        tree_model: PerClassDecisionTreeModel,
        weak_supervised_model: MLP,
        tree_weight: float = 0.5,
        nn_weight: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize the hybrid ensemble model.

        Args:
            tree_model: Trained PerClassDecisionTreeModel instance
            weak_supervised_model: Trained MLP model from weak supervision
            tree_weight: Weight for tree ensemble predictions (default: 0.5)
            nn_weight: Weight for neural network predictions (default: 0.5)
            device: Device to run the neural network on
        """
        self.tree_model = tree_model
        self.weak_supervised_model = weak_supervised_model
        self.tree_weight = tree_weight
        self.nn_weight = nn_weight
        self.device = device

        # Get classes from tree model
        self.classes = self.tree_model.classes
        self.n_classes = len(self.classes)

        print(f"Hybrid ensemble initialized with {self.n_classes} classes")
        print(f"Tree weight: {self.tree_weight}, NN weight: {self.nn_weight}")

    def _get_tree_votes(self, X: np.ndarray) -> np.ndarray:
        """
        Get voting fractions from decision trees for each class.

        Args:
            X: Input features

        Returns:
            Array of shape (n_samples, n_classes) with voting fractions
        """
        n_samples = X.shape[0]
        tree_votes = np.zeros((n_samples, self.n_classes))

        # Collect predictions from all trees for all classes
        all_predictions = []

        for class_id in self.classes:
            for tree_idx in range(len(self.tree_model.trees_per_class[class_id])):
                predictions, _ = self.tree_model.predict_single_tree(
                    X, class_id, tree_idx
                )
                # Convert to class labels: 1 -> class_id, 0 -> -1 (not this class), -1 -> -1 (abstain)
                class_predictions = np.full(n_samples, -1, dtype=int)
                class_predictions[predictions == 1] = class_id
                all_predictions.append(class_predictions)

        all_predictions = np.array(all_predictions)  # Shape: (n_trees, n_samples)

        # Calculate voting fractions for each sample and class
        for i in range(n_samples):
            votes = all_predictions[:, i]
            # Remove abstentions (-1) from voting
            valid_votes = votes[votes != -1]

            if len(valid_votes) > 0:
                # Count votes for each class
                for class_idx, class_id in enumerate(self.classes):
                    tree_votes[i, class_idx] = np.sum(valid_votes == class_id) / len(
                        valid_votes
                    )
            # If no valid votes, tree_votes[i, :] remains zeros

        return tree_votes

    def _get_nn_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Get softmax probabilities from the weak supervised neural network.

        Args:
            X: Input features

        Returns:
            Array of shape (n_samples, n_classes) with softmax probabilities
        """
        self.weak_supervised_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X).to(self.device)
            logits = self.weak_supervised_model(X_tensor)
            probabilities = F.softmax(logits, dim=-1)
            return probabilities.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the hybrid ensemble voting mechanism.

        Args:
            X: Input features

        Returns:
            Predicted class labels
        """
        # Get tree voting fractions
        tree_votes = self._get_tree_votes(X)

        # Get neural network probabilities
        nn_probs = self._get_nn_probabilities(X)

        # Combine predictions using weighted average
        combined_votes = self.tree_weight * tree_votes + self.nn_weight * nn_probs

        # Return class with highest vote
        predictions = np.argmax(combined_votes, axis=1)

        # Map back to original class labels
        return np.array([self.classes[pred] for pred in predictions])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability scores for each class using the hybrid ensemble.

        Args:
            X: Input features

        Returns:
            Array of shape (n_samples, n_classes) with probability scores
        """
        # Get tree voting fractions
        tree_votes = self._get_tree_votes(X)

        # Get neural network probabilities
        nn_probs = self._get_nn_probabilities(X)

        # Combine predictions using weighted average
        combined_votes = self.tree_weight * tree_votes + self.nn_weight * nn_probs

        return combined_votes

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the hybrid ensemble on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate per-class accuracy
        per_class_accuracy = {}
        for class_idx, class_id in enumerate(self.classes):
            class_mask = y_test == class_id
            if np.sum(class_mask) > 0:
                per_class_accuracy[class_id] = accuracy_score(
                    y_test[class_mask], y_pred[class_mask]
                )
            else:
                per_class_accuracy[class_id] = 0.0

        results = {
            "accuracy": accuracy,
            "per_class_accuracy": per_class_accuracy,
            "predictions": y_pred,
            "probabilities": y_proba,
            "true_labels": y_test,
            "tree_weight": self.tree_weight,
            "nn_weight": self.nn_weight,
        }

        return results

    def get_component_predictions(self, X: np.ndarray) -> Dict:
        """
        Get predictions from individual components for analysis.

        Args:
            X: Input features

        Returns:
            Dictionary with predictions from each component
        """
        tree_votes = self._get_tree_votes(X)
        nn_probs = self._get_nn_probabilities(X)

        # Get tree predictions (majority vote)
        tree_predictions = np.argmax(tree_votes, axis=1)
        tree_predictions = np.array([self.classes[pred] for pred in tree_predictions])

        # Get NN predictions
        nn_predictions = np.argmax(nn_probs, axis=1)
        nn_predictions = np.array([self.classes[pred] for pred in nn_predictions])

        return {
            "tree_votes": tree_votes,
            "nn_probabilities": nn_probs,
            "tree_predictions": tree_predictions,
            "nn_predictions": nn_predictions,
        }

    def analyze_agreement(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Analyze agreement between tree and neural network predictions.

        Args:
            X: Input features
            y: True labels

        Returns:
            Dictionary with agreement analysis
        """
        component_preds = self.get_component_predictions(X)
        tree_preds = component_preds["tree_predictions"]
        nn_preds = component_preds["nn_predictions"]
        hybrid_preds = self.predict(X)

        # Calculate agreement rates
        tree_nn_agreement = np.mean(tree_preds == nn_preds)
        tree_hybrid_agreement = np.mean(tree_preds == hybrid_preds)
        nn_hybrid_agreement = np.mean(nn_preds == hybrid_preds)

        # Calculate individual accuracies
        tree_accuracy = accuracy_score(y, tree_preds)
        nn_accuracy = accuracy_score(y, nn_preds)
        hybrid_accuracy = accuracy_score(y, hybrid_preds)

        # Find samples where components disagree
        disagreement_mask = tree_preds != nn_preds
        disagreement_indices = np.where(disagreement_mask)[0]

        # Analyze disagreement cases
        disagreement_analysis = {
            "tree_correct": 0,
            "nn_correct": 0,
            "both_wrong": 0,
            "hybrid_correct": 0,
        }

        for idx in disagreement_indices:
            tree_correct = tree_preds[idx] == y[idx]
            nn_correct = nn_preds[idx] == y[idx]
            hybrid_correct = hybrid_preds[idx] == y[idx]

            if tree_correct and not nn_correct:
                disagreement_analysis["tree_correct"] += 1
            elif nn_correct and not tree_correct:
                disagreement_analysis["nn_correct"] += 1
            else:
                disagreement_analysis["both_wrong"] += 1

            if hybrid_correct:
                disagreement_analysis["hybrid_correct"] += 1

        # Convert counts to percentages
        n_disagreements = len(disagreement_indices)
        if n_disagreements > 0:
            for key in disagreement_analysis:
                disagreement_analysis[key] = (
                    disagreement_analysis[key] / n_disagreements
                )

        return {
            "tree_nn_agreement": tree_nn_agreement,
            "tree_hybrid_agreement": tree_hybrid_agreement,
            "nn_hybrid_agreement": nn_hybrid_agreement,
            "tree_accuracy": tree_accuracy,
            "nn_accuracy": nn_accuracy,
            "hybrid_accuracy": hybrid_accuracy,
            "disagreement_rate": 1 - tree_nn_agreement,
            "disagreement_analysis": disagreement_analysis,
            "n_disagreements": n_disagreements,
        }

    def set_weights(self, tree_weight: float, nn_weight: float):
        """
        Update the weights for the hybrid ensemble.

        Args:
            tree_weight: Weight for tree ensemble predictions
            nn_weight: Weight for neural network predictions
        """
        total_weight = tree_weight + nn_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        self.tree_weight = tree_weight
        self.nn_weight = nn_weight
        print(f"Updated weights - Tree: {self.tree_weight}, NN: {self.nn_weight}")

    def save_model(self, filepath: str):
        """
        Save the hybrid ensemble model.

        Args:
            filepath: Path to save the model
        """
        model_state = {
            "tree_model": self.tree_model,
            "weak_supervised_model_state": self.weak_supervised_model.state_dict(),
            "tree_weight": self.tree_weight,
            "nn_weight": self.nn_weight,
            "classes": self.classes,
            "n_classes": self.n_classes,
        }
        torch.save(model_state, filepath)
        print(f"Model saved to {filepath}")
