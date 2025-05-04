import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def predict_proba_ordered(probs, classes_, all_classes):
    """
    Reorder probability columns to match full set of classes.
    probs: array (n_samples, n_seen_classes)
    classes_: array of seen class labels
    all_classes: array of all possible class labels
    """
    n_samples = probs.shape[0]
    ordered = np.zeros((n_samples, all_classes.size))
    for idx, cls in enumerate(all_classes):
        if cls in classes_:
            cls_index = np.where(classes_ == cls)[0][0]
            ordered[:, idx] = probs[:, cls_index]
    return ordered

def train_model(X_train, y_train, random_state=0):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = MLPClassifier(
        hidden_layer_sizes=(50,),
        activation='tanh', solver='lbfgs',
        max_iter=5000, random_state=random_state
    )
    clf.fit(Xs, y_train)
    return clf, scaler

def my_scores_with_scaler(X_test, y_test, scaler, clf, all_classes):
    Xs = scaler.transform(X_test)
    y_pred = clf.predict(Xs)
    probs = clf.predict_proba(Xs)
    # reorder to full class list
    probs_ord = predict_proba_ordered(probs, clf.classes_, all_classes)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', labels=all_classes)
    
    if len(all_classes) ==2 :            
        f1 = f1_score(y_test, y_pred, pos_label= all_classes[1])
        auc = roc_auc_score(y_test, probs_ord[:,1], labels = all_classes)
    else : 
        f1 = f1_score(y_test, y_pred, average='weighted', labels = all_classes)
        auc = roc_auc_score(y_test, probs_ord, multi_class='ovr', average='weighted', labels = all_classes)
    
    return acc, f1, auc