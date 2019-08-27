
def normalize(df, features):
    for f in features:
        range = df[f].max() - df[f].min()
        mean = df[f].mean()
        df[f] = (df[f] - mean) / range
