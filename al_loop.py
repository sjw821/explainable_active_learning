import numpy as np
from sklearn.model_selection import train_test_split
from data_utils import dataset_load_all, noise_injection_uni_simple
from model_utils import train_model, my_scores_with_scaler
from explainer_utils import get_acquisition_scores

def run_active_learning(dataname, method, rep, lambda_val=0.0, n_iter=170, test_size=0.5):
    """
    method: 'al' for conventional active learning,
            'al_ei' for explainable intervention
    lambda_val: weight for noisy features when using 'al_ei'
    """
    # Load and split dataset
    X, y = dataset_load_all(dataname, seed=rep)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=rep
    )

    # Inject noise into training set
    X_train_noisy, noise_idx = noise_injection_uni_simple(X_train, seed=rep)

    # Initialize labeled/unlabeled pools
    n_init, budget = 30, 200
    np.random.seed(rep)
    idxs = np.arange(len(X_train_noisy))
    init_idx = np.random.choice(idxs, n_init, replace=False)
    L_idx = list(init_idx)
    U_idx = list(set(idxs) - set(L_idx))

    # Collect metrics
    metrics = []
    for i in range(n_iter):
        # Train model
        clf, scaler = train_model(
            X_train_noisy[L_idx], y_train[L_idx], random_state=rep+i
        )
        # Evaluate on test set
        acc, f1, auc = my_scores_with_scaler(
            X_test, y_test, scaler, clf, np.unique(y)
        )
        metrics.append({'accuracy': acc, 'f1': f1, 'auroc': auc})

        if len(L_idx) >= budget:
            break

        # Determine acquisition method
        if method == 'al':
            lambda_param = 1.0
        elif method == 'al_ei':
            lambda_param = lambda_val
        else:
            raise ValueError("method must be 'al' or 'al_ei'.")

        # Compute acquisition scores
        scores = get_acquisition_scores(
            clf, scaler,
            X_train_noisy[U_idx],
            method=method,
            noise_idx=noise_idx,
            lambda_val=lambda_param,
            rep=rep+i
        )
        # Select next instance
        sel = np.argmax(scores)
        L_idx.append(U_idx[sel])
        del U_idx[sel]

    return metrics