# Hybrid Ensemble Model

This module implements a hybrid ensemble model that combines decision tree predictions with weak supervised classifier predictions using the voting mechanism:

```
votes[l] = 0.5 * soft-max-prob(weak-supervised-classifier, x) + 0.5 * #(trees that vote for l on input x)/#(trees that do not abstain on x)
prediction for input x: argmax(votes)
```

## Overview

The `HybridEnsembleModel` combines two complementary approaches:

1. **Decision Tree Ensemble**: Uses `PerClassDecisionTreeModel` to create multiple decision trees per class with abstention capability
2. **Weak Supervised Neural Network**: Uses an MLP trained with weak supervision techniques from noisy tree predictions

## Key Features

- **Hybrid Voting**: Combines tree ensemble votes with neural network probabilities
- **Configurable Weights**: Adjustable weights for tree vs. neural network predictions
- **Abstention Handling**: Properly handles tree abstentions in voting calculations
- **Analysis Tools**: Built-in methods to analyze component agreement and performance
- **Model Persistence**: Save and load trained ensemble models

## Usage

### Basic Usage

```python
from cawl.models.ensemble import create_hybrid_ensemble, HybridEnsembleModel

# Create and train the ensemble
ensemble = create_hybrid_ensemble(
    tree_model=tree_model,  # Pre-trained PerClassDecisionTreeModel
    X_train=X_train,
    l_train=l_train,  # Weak labels from trees
    eta=eta,  # Error rates for weak labelers
    input_dim=X_train.shape[1],
    num_classes=n_classes,
    tree_weight=0.5,
    nn_weight=0.5,
    device="cpu"
)

# Make predictions
predictions = ensemble.predict(X_test)
probabilities = ensemble.predict_proba(X_test)

# Evaluate performance
results = ensemble.evaluate(X_test, y_test)
print(f"Accuracy: {results['accuracy']:.4f}")
```

### From Existing Models

```python
# If you already have trained models
ensemble = HybridEnsembleModel(
    tree_model=tree_model,
    weak_supervised_model=nn_model,
    tree_weight=0.5,
    nn_weight=0.5,
    device="cpu"
)
```

### Analysis and Debugging

```python
# Get predictions from individual components
component_preds = ensemble.get_component_predictions(X_test)

# Analyze agreement between components
agreement_analysis = ensemble.analyze_agreement(X_test, y_test)
print(f"Tree-NN agreement: {agreement_analysis['tree_nn_agreement']:.4f}")
print(f"Disagreement rate: {agreement_analysis['disagreement_rate']:.4f}")
```

### Model Persistence

```python
# Save model
ensemble.save_model("hybrid_ensemble.pth")

# Load model
loaded_ensemble = HybridEnsembleModel.load_model("hybrid_ensemble.pth", device="cpu")
```

## API Reference

### HybridEnsembleModel

#### Constructor

```python
HybridEnsembleModel(
    tree_model: PerClassDecisionTreeModel,
    weak_supervised_model: MLP,
    tree_weight: float = 0.5,
    nn_weight: float = 0.5,
    device: str = "cpu"
)
```

#### Methods

- `predict(X)`: Make class predictions
- `predict_proba(X)`: Get probability scores for all classes
- `evaluate(X_test, y_test)`: Evaluate model performance
- `get_component_predictions(X)`: Get predictions from individual components
- `analyze_agreement(X, y)`: Analyze agreement between components
- `set_weights(tree_weight, nn_weight)`: Update ensemble weights
- `save_model(filepath)`: Save model to file
- `load_model(filepath, device)`: Load model from file (class method)

### create_hybrid_ensemble

```python
create_hybrid_ensemble(
    tree_model: PerClassDecisionTreeModel,
    X_train: np.ndarray,
    l_train: np.ndarray,
    eta: torch.Tensor,
    input_dim: int,
    num_classes: int,
    tree_weight: float = 0.5,
    nn_weight: float = 0.5,
    device: str = "cpu",
    train_epochs: int = 1000,
    **nn_kwargs
) -> HybridEnsembleModel
```

## Voting Mechanism Details

The ensemble uses a weighted combination of two prediction sources:

1. **Tree Voting Fraction**: `#(trees that vote for l on input x)/#(trees that do not abstain on x)`

   - Only considers trees that didn't abstain (were confident enough to make predictions)
   - Normalizes by the total number of non-abstaining trees

2. **Neural Network Probabilities**: `soft-max-prob(weak-supervised-classifier, x)`
   - Softmax probabilities from the trained neural network
   - Trained using weak supervision with noisy labels from trees

The final prediction is: `argmax(tree_weight * tree_votes + nn_weight * nn_probs)`

## Example Workflow

1. **Train Decision Trees**: Use `PerClassDecisionTreeModel` to train multiple trees per class
2. **Generate Weak Labels**: Get predictions from all trees (including abstentions)
3. **Calculate Error Rates**: Estimate error rates for each tree/weak labeler
4. **Train Neural Network**: Use weak supervision to train an MLP
5. **Create Ensemble**: Combine both models with the hybrid voting mechanism
6. **Evaluate and Analyze**: Use built-in analysis tools to understand performance

## Advantages

- **Robustness**: Combines interpretable tree decisions with flexible neural network patterns
- **Handles Uncertainty**: Tree abstentions help identify uncertain cases
- **Weak Supervision**: Can train neural networks without perfect labels
- **Configurable**: Adjustable weights allow tuning the ensemble behavior
- **Analyzable**: Built-in tools to understand component contributions

## Requirements

- `torch`
- `numpy`
- `sklearn`
- `cawl.models.tree.per_class_decision_tree_model`
- `cawl.models.wl.wl`
