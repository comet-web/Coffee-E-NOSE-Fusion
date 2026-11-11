"""Classical classifiers used across experiments (hyperparameters fixed in advance,
except the SVM whose C is tuned by an inner grid search = nested CV)."""
from __future__ import annotations

import warnings

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def make_clf(name: str, seed: int = 0, balanced: bool = True):
    cw = "balanced" if balanced else None
    if name == "LR":
        return LogisticRegression(C=1.0, max_iter=5000, class_weight=cw)
    if name == "SVM":
        svc = SVC(kernel="rbf", gamma="scale", probability=True, class_weight=cw, random_state=seed)
        return GridSearchCV(svc, {"C": [1, 10, 100]}, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                            scoring="balanced_accuracy", n_jobs=1)
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, class_weight=cw, random_state=seed, n_jobs=2)
    if name == "kNN":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "MLP":  # same size as the paper's network: 1 hidden layer, 10 units
        return MLPClassifier(hidden_layer_sizes=(10,), solver="adam", max_iter=3000, random_state=seed)
    if name == "GB":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                              class_weight=cw, random_state=seed)
    raise ValueError(name)


def scaled(name: str, seed: int = 0, pca: int | float | None = None, balanced: bool = True):
    """StandardScaler (+ optional PCA) + classifier, all fitted inside the training fold."""
    steps = [StandardScaler()]
    if pca is not None:
        steps.append(PCA(n_components=pca, random_state=seed))
    steps.append(make_clf(name, seed, balanced))
    return make_pipeline(*steps)


S1_CLFS = ["LR", "SVM", "RF", "kNN", "MLP"]
