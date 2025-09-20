from sklearn.ensemble import HistGradientBoostingClassifier

def get_histgb_model(random_seed=42):
    """返回一个 HistGB 分类器"""
    clf = HistGradientBoostingClassifier(
        max_iter=400,
        random_state=random_seed,
        learning_rate=0.05,
        max_depth=None
    )
    return clf
