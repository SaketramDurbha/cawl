"""
Example usage of the HybridEnsembleModel.

This script demonstrates how to:
1. Train a PerClassDecisionTreeModel
2. Create weak labels from the tree model
3. Train a weak supervised neural network
4. Create and use the hybrid ensemble
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from cawl.models.ensemble import HybridEnsembleModel
from cawl.models.tree.per_class_decision_tree_model import PerClassDecisionTreeModel


def example_usage():
    """Example of how to use the hybrid ensemble model."""

    # This is a placeholder for the actual dataset
    # In practice, you would load your dataset here
    print("Note: This is an example script. You need to provide actual data.")
    print("Replace the placeholder data with your actual dataset.")

    # Placeholder data (replace with actual data)
    # X_train, y_train = load_your_dataset()
    # X_test, y_test = load_your_test_dataset()

    # For demonstration, we'll create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    n_classes = 5

    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, n_classes, n_samples)

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Number of classes: {n_classes}")

    # Step 1: Create and train the decision tree model
    print("\n=== Step 1: Training Decision Tree Model ===")

    # Note: In practice, you would pass an actual dataset object
    # tree_model = PerClassDecisionTreeModel(dataset, max_depth=3, n_trees_per_class=5)
    # tree_model.train_all_trees()

    # For this example, we'll create a mock tree model
    # (In practice, you would use the real implementation)
    print("Decision tree model training would happen here...")

    # Step 2: Generate weak labels from the tree model
    print("\n=== Step 2: Generating Weak Labels ===")

    # In practice, you would do:
    # l_train = tree_model.predict_ensemble(X_train, method=None)
    # This gives you predictions from each tree (shape: [n_samples, n_trees])

    # For this example, we'll create mock weak labels
    n_trees = 25  # 5 trees per class * 5 classes
    l_train = np.random.randint(-1, n_classes, size=(len(X_train), n_trees))
    print(f"Weak labels shape: {l_train.shape}")

    # Step 3: Calculate error rates for weak labelers
    print("\n=== Step 3: Calculating Error Rates ===")

    # In practice, you would calculate error rates based on validation performance
    # eta = torch.tensor([1 - accuracy for accuracy in tree_accuracies])

    # For this example, we'll use random error rates
    eta = torch.tensor(np.random.uniform(0.1, 0.4, n_trees))
    print(f"Error rates shape: {eta.shape}")

    # Step 4: Create and train the hybrid ensemble
    print("\n=== Step 4: Creating Hybrid Ensemble ===")

    # In practice, you would do:
    # ensemble = create_hybrid_ensemble(
    #     tree_model=tree_model,
    #     X_train=X_train,
    #     l_train=l_train,
    #     eta=eta,
    #     input_dim=X_train.shape[1],
    #     num_classes=n_classes,
    #     tree_weight=0.5,
    #     nn_weight=0.5,
    #     device="cpu",
    #     train_epochs=1000
    # )

    print("Hybrid ensemble creation would happen here...")

    # Step 5: Evaluate the ensemble
    print("\n=== Step 5: Evaluating the Ensemble ===")

    # In practice, you would do:
    # results = ensemble.evaluate(X_val, y_val)
    # print(f"Hybrid ensemble accuracy: {results['accuracy']:.4f}")

    print("Ensemble evaluation would happen here...")

    # Step 6: Analyze component agreement
    print("\n=== Step 6: Analyzing Component Agreement ===")

    # In practice, you would do:
    # agreement_analysis = ensemble.analyze_agreement(X_val, y_val)
    # print(f"Tree-NN agreement rate: {agreement_analysis['tree_nn_agreement']:.4f}")
    # print(f"Disagreement rate: {agreement_analysis['disagreement_rate']:.4f}")

    print("Component agreement analysis would happen here...")

    print("\n=== Example Complete ===")
    print("To use this in practice:")
    print("1. Replace placeholder data with your actual dataset")
    print("2. Uncomment the actual implementation code")
    print("3. Adjust hyperparameters as needed")


def create_hybrid_ensemble_from_existing_models(
    tree_model: PerClassDecisionTreeModel,
    weak_supervised_model: torch.nn.Module,
    tree_weight: float = 0.5,
    nn_weight: float = 0.5,
    device: str = "cpu",
) -> HybridEnsembleModel:
    """
    Create a hybrid ensemble from existing trained models.

    Args:
        tree_model: Pre-trained PerClassDecisionTreeModel
        weak_supervised_model: Pre-trained neural network
        tree_weight: Weight for tree predictions
        nn_weight: Weight for neural network predictions
        device: Device to run on

    Returns:
        HybridEnsembleModel instance
    """

    ensemble = HybridEnsembleModel(
        tree_model=tree_model,
        weak_supervised_model=weak_supervised_model,
        tree_weight=tree_weight,
        nn_weight=nn_weight,
        device=device,
    )

    return ensemble


if __name__ == "__main__":
    example_usage()
