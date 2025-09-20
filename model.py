from sklearn.ensemble import HistGradientBoostingClassifier

def get_histgb_model(random_seed=42):
    """Only for classification"""
    clf = HistGradientBoostingClassifier(
        max_iter=400,
        random_state=random_seed,
        learning_rate=0.05,
        max_depth=None
    )
    return clf
