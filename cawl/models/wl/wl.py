import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()

        # More complex architecture with regularization
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Add batch normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))  # Add dropout for regularization
            current_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(current_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


# Loss function
@torch.no_grad()
def projection(logits_f, l, eta):
    """logits_f: logits from the model (shape: [batch_size, num_classes])
    l: weak labels (shape: [batch_size, num_labelers]), -1 means abstain
    eta: error rates for each labeler (shape: [num_labelers, 1])

    Returns:
    prob_q: projected probabilities (shape: [batch_size, num_classes])
    projection_loss: loss for the projection step"""

    prob_f = F.softmax(logits_f, dim=-1)
    num_labelers = l.shape[1]
    num_classes = prob_f.shape[1]

    # calculate the lambdas for each labeler (equation 24)
    lambdas = []
    for j in range(num_labelers):
        non_abstain_mask = l[:, j] != -1

        z_j = torch.gather(
            input=prob_f[non_abstain_mask],
            dim=-1,
            index=l[:, j][non_abstain_mask].reshape(-1, 1),
        )
        lambda_j = torch.log(
            1 + max(0, 1 - eta[j] - z_j.mean()) / (z_j * (1 - z_j)).mean()
        ).reshape(-1)
        lambdas.append(lambda_j)
    lambdas = torch.stack(lambdas, dim=-1)

    # project prob_f onto the set of soft labels that satisfy the constraints (equation 25)
    update = []
    for k in range(num_classes):
        update.append(((l == k) * lambdas).sum(dim=1))
    update = torch.exp(torch.stack(update, dim=1))

    prob_q_unnorm = prob_f * update
    prob_q = prob_q_unnorm / prob_q_unnorm.sum(dim=-1, keepdim=True)

    # calculate the constraints violation as well
    dist_q_l = []
    for j in range(num_labelers):
        non_abstain_mask = l[:, j] != -1
        dist_q_l_j = (
            1
            - torch.gather(
                input=prob_q[non_abstain_mask],
                dim=-1,
                index=l[:, j][non_abstain_mask].reshape(-1, 1),
            ).mean()
        )
        dist_q_l.append(dist_q_l_j)
    dist_q_l = torch.stack(dist_q_l, dim=-1)
    constraints_violation = torch.clamp(dist_q_l - eta.reshape(-1), min=0)

    return prob_q, dist_q_l, constraints_violation


# Training
def train(model, X_train, l_train, eta, configs, X_val=None, y_val=None):
    """Trains the model with the given configurations and data.
    model: the model to train
    X_train: training features
    l_train: weak labels for training
    eta: error rates for each labeler
    configs: dictionary with training configurations
    X_val: validation features (optional)
    y_val: validation labels (optional)

    If X_val and y_val are provided, the model will be evaluated on the validation set after each epoch and we will use early stopping based on validation loss.

    Returns:
    model: trained model
    """

    device = next(model.parameters()).device

    X_train, l_train, eta = X_train.to(device), l_train.to(device), eta.to(device)
    has_validation = X_val is not None and y_val is not None
    if has_validation:
        X_val, y_val = X_val.to(device), y_val.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"]
    )
    model.train()
    best_val_loss = float("inf")

    for epoch in range(configs["epochs"]):
        optimizer.zero_grad()
        logits_f = model(X_train)
        prob_q, dist_q_l, constraints_violation = projection(logits_f, l_train, eta)
        loss = F.cross_entropy(input=logits_f, target=prob_q, reduction="mean")
        loss.backward()
        optimizer.step()

        if has_validation:
            val_accuracy, val_loss = eval_model(model, X_val, y_val)
            print(
                f'Epoch {epoch+1}/{configs["epochs"]}, Loss: {loss.item():.4f}, Constraints Violation: {constraints_violation.mean().item():.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}'
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
        else:
            print(
                f'Epoch {epoch+1}/{configs["epochs"]}, Loss: {loss.item():.4f}',
                f"Constraints Violation: {constraints_violation.mean().item():.4f}",
            )

    if has_validation:
        model.load_state_dict(best_model_state)
        print(
            "Best validation loss:",
            best_val_loss,
            "Validation accuracy:",
            eval_model(model, X_val, y_val)[0],
        )
    return model, best_val_loss


# Evaluation
@torch.no_grad()
def eval_model(model, X, y):
    model.eval()
    device = next(model.parameters()).device
    X, y = X.to(device), y.to(device)
    logits = model(X)
    prob_f = F.softmax(logits, dim=-1)
    predictions = torch.argmax(prob_f, dim=-1)
    accuracy = 100 * (predictions == y).float().mean().item()
    loss = F.cross_entropy(logits, y)
    return accuracy, loss.item()


@torch.no_grad()
def eval_majority_vote(l, y):
    """Evaluate the majority vote accuracy,
    l: weak labels (shape: [batch_size, num_labelers]), -1 means abstain
    y: true labels (shape: [batch_size])

    Returns:
    mv_accuracy: average majority vote accuracy, ties are broken randomly"""
    mv = ((l != -1) * l).sum(dim=1) / (l != -1).sum(dim=1)
    valid_point = ~torch.isnan(mv)

    mv_accuracy0 = (
        ((mv[valid_point] > 0.5).to(torch.long) == y[valid_point])
        .to(torch.float)
        .mean()
        .item()
    )
    mv_accuracy1 = (
        ((mv[valid_point] >= 0.5).to(torch.long) == y[valid_point])
        .to(torch.float)
        .mean()
        .item()
    )

    return 100 * (mv_accuracy0 + mv_accuracy1) / 2
