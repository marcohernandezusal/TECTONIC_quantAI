import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_split_data(filepath, target='Corrosion', test_size=0.2, random_state=21, random=True):
    df = pd.read_csv(filepath, index_col=False, header=0)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the dataset.")
    X = df.drop(columns=[target])
    y = df[target]
    if random:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    else:
        # If not random, we assume the data is already ordered and we split sequentially
        # This is useful for time series or ordered data
        X_train = X.iloc[:-int(len(X) * test_size)]
        y_train = y.iloc[:-int(len(y) * test_size)]
        X_test = X.iloc[-int(len(X) * test_size):]
        y_test = y.iloc[-int(len(y) * test_size):]
        return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def get_labeled_unlabeled_split(X, y, labeled_fraction=0.1, random_state=42):
    """
    Splits X,y into a small labeled set and a large unlabeled pool.
    unlabeled set returns X_unlab only (y unknown).
    """
    if not 0 < labeled_fraction < 1:
        raise ValueError("labeled_fraction must be between 0 and 1.")
    n_total = len(X)
    n_lab = max(1, int(n_total * labeled_fraction))
    # stratify if classification; here regression: random sample
    X_lab = X.sample(n=n_lab, random_state=random_state)
    y_lab = y.loc[X_lab.index]
    X_unlab = X.drop(index=X_lab.index)
    return X_lab.reset_index(drop=True), y_lab.reset_index(drop=True), X_unlab.reset_index(drop=True)
