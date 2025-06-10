
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.inspection import permutation_importance
from models.ranknet_bidirectional import load_model as load_bi_model
from models.ranknet_unidirectional import load_model as load_uni_model
from utils.preprocess import prepare_vectors
import torch

def evaluate_all(df, version="bi"):
    user_df = df.sample(1).reset_index(drop=True)
    user_df.columns = ['name', 'age_month', 'weight', 'color', 'personality', 'region', 'vaccinated', 
                       'naver', 'lat', 'lon', 'pref_age', 'pref_weight', 'pref_color',
                       'pref_personality', 'pref_region', 'pref_vaccine'][:len(df.columns)]

    _, vectors, _, _, _ = prepare_vectors(user_df, df, mode=version)
    model = load_bi_model() if version == "bi" else load_uni_model()

    with torch.no_grad():
        scores = model(torch.tensor(vectors, dtype=torch.float32)).squeeze().numpy()

    # permutation importance (surrogate model 기반)
    from sklearn.ensemble import RandomForestRegressor
    X = vectors
    y = scores
    surrogate = RandomForestRegressor().fit(X, y)
    r = permutation_importance(surrogate, X, y, n_repeats=10, random_state=0)

    result = {
        "top_features": r.importances_mean.argsort()[::-1][:5].tolist(),
        "mean_importance": r.importances_mean.tolist()
    }
    return result
