
def normalize(df, features):
    for f in features:
        range = df[f].max() - df[f].min()
        df[f] = df[f] / range
