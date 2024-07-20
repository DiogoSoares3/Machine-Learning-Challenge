import pandas as pd
import numpy as np


def label_transform(y: np.array) -> pd.Series:
    y = pd.Series(y)
    return y.replace({'neg': 0, 'pos': 1})
