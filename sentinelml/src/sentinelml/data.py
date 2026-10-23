import pandas as pd
from sklearn.datasets import make_classification
from sentinelml.config import FEATURE_NAMES, RANDOM_STATE, TARGET_NAME

def generate_dataset(n_samples: int = 1800, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples, n_features=len(FEATURE_NAMES), n_informative=6,
        n_redundant=2, n_repeated=0, n_classes=2, weights=[0.58, 0.42],
        class_sep=1.0, random_state=random_state,
    )
    frame = pd.DataFrame(X, columns=FEATURE_NAMES)
    frame[TARGET_NAME] = y.astype(int)
    return frame
