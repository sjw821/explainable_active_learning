import numpy as np
import shap

class UncertaintyScorer:
    def __init__(self, model, method, scaler=None):
        self.model = model
        self.method = method
        self.scaler = scaler

    def predict(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        probs = self.model.predict_proba(X)
        s = np.sort(probs, axis=1)
        return -(s[:, -1] - s[:, -2])  # margin


def get_acquisition_scores(
    clf, scaler, X_pool, method,
    noise_idx=None, lambda_val=1.0, rep=0
):
    """
    Returns acquisition scores for 'al' (uncertainty) or 'al_ei' (feature-weighted uncertainty).
    """
    # Base uncertainty scores
    scorer = UncertaintyScorer(clf, 'margin', scaler)
    base_scores = scorer.predict(X_pool)
    # Conventional AL: return raw uncertainty
    if method == 'al':
        return base_scores

    # AL_EI: adjust scores via SHAP
    # select top-k uncertain instances
    topk = np.argsort(base_scores)[-50:]
    # build explainer on the pool
    explainer = shap.PermutationExplainer(
        scorer.predict,
        shap.sample(X_pool, 100, random_state=rep)
    )
    # compute shap values for topk
    shap_vals = explainer.shap_values(X_pool[topk])  # array (topk, n_features)
    # define feature weights
    weights = np.ones(X_pool.shape[1])
    if noise_idx is not None:
        weights[noise_idx] = lambda_val
    # compute adjusted scores for topk
    adjusted = np.sum(shap_vals * weights, axis=1)
    # map back to full pool
    scores = np.zeros_like(base_scores)
    for idx, val in zip(topk, adjusted):
        scores[idx] = val
    return scores
