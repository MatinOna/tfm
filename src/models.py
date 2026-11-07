import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.inspection import permutation_importance

SEED = 42
N_SPLITS = 5
N_REPEATS = 10
SCORING = "f1_macro"


def make_cv(n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS, seed: int = SEED):
    """Stratified k-fold cross-validation repeated with different shuffles."""
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def build_pipeline(estimator, scale: bool) -> Pipeline:
    """Optional standardisation followed by the model, fitted together inside each fold."""
    steps = [("scaler", StandardScaler())] if scale else []
    steps.append(("model", estimator))
    return Pipeline(steps)

def tune(pipeline: Pipeline, param_grid, X, y, cv=None, **fit_params) -> GridSearchCV:
    """Try every combination of param_grid with repeated cross-validation and keep the best (macro-F1)."""
    search = GridSearchCV(pipeline, param_grid, scoring=SCORING, cv=cv or make_cv(), n_jobs=-1)
    search.fit(X, y, **fit_params)
    return search


def best_fold_scores(search: GridSearchCV):
    """Macro-F1 of the selected configuration in each of the cross-validation rounds."""
    i = search.best_index_
    n_rounds = search.n_splits_
    return np.array([search.cv_results_[f"split{k}_test_score"][i] for k in range(n_rounds)])

def predict_genres(model, X, genres):
    """Predict genre names; models trained with numeric labels (XGBoost) are mapped back to names."""
    pred = np.asarray(model.predict(X))
    if np.issubdtype(pred.dtype, np.number):
        pred = np.asarray(genres)[pred.astype(int)]
    return pred

def cv_predictions(model, X, y, genres, numeric_labels=False, sample_weight=None, cv=None) -> pd.DataFrame:
    """Out-of-fold genre predictions of each repetition of the cross-validation.

    Returns one row per interpretation and one column per repetition: in every repetition each
    interpretation is predicted once, by the model trained without its group.
    """
    cv = cv or make_cv()
    n_splits = cv.cvargs["n_splits"]
    y_fit = y.map({g: i for i, g in enumerate(genres)}) if numeric_labels else y
    preds = pd.DataFrame(index=X.index, columns=range(cv.n_repeats), dtype=object)
    for k, (train_idx, eval_idx) in enumerate(cv.split(X, y)):
        fitted = clone(model)
        fit_params = {"model__sample_weight": sample_weight[train_idx]} if sample_weight is not None else {}
        fitted.fit(X.iloc[train_idx], y_fit.iloc[train_idx], **fit_params)
        preds.iloc[eval_idx, k // n_splits] = predict_genres(fitted, X.iloc[eval_idx], genres)
    return preds

def grouped_cv_splits(X, y, groups, n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS, seed: int = SEED):
    """Repeated stratified cross-validation keeping each group (e.g. a recording session) in a single fold."""
    splits = []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        splits += list(cv.split(X, y, groups))
    return splits


def cv_permutation_importance(model, X, y, genres, numeric_labels=False, sample_weight=None,
                              n_permutations: int = 10, seed: int = SEED) -> pd.Series:
    """Permutation importance of each feature: mean drop in macro-F1 when its values are shuffled.

    Computed on the evaluation fold of each round of one cross-validation repetition (5 rounds)
    and averaged over the rounds.
    """
    cv = make_cv(n_repeats=1, seed=seed)
    y_fit = y.map({g: i for i, g in enumerate(genres)}) if numeric_labels else y
    importances = []
    for train_idx, eval_idx in cv.split(X, y):
        fitted = clone(model)
        fit_params = {"model__sample_weight": sample_weight[train_idx]} if sample_weight is not None else {}
        fitted.fit(X.iloc[train_idx], y_fit.iloc[train_idx], **fit_params)
        result = permutation_importance(fitted, X.iloc[eval_idx], y_fit.iloc[eval_idx], scoring=SCORING,
                                        n_repeats=n_permutations, random_state=seed, n_jobs=-1)
        importances.append(result.importances_mean)
    return pd.Series(np.mean(importances, axis=0), index=X.columns)


def corrected_ttest(scores_a, scores_b, test_train_ratio: float = 1 / (N_SPLITS - 1)):
    """Corrected t-test of Nadeau and Bengio (2003) on the paired cross-validation scores of two models.

    Returns the mean difference (a - b), the t statistic and the two-sided p-value.
    """
    diff = np.asarray(scores_a) - np.asarray(scores_b)
    n = len(diff)
    t = diff.mean() / np.sqrt((1 / n + test_train_ratio) * diff.var(ddof=1))
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return diff.mean(), t, p
