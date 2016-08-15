def normalize_dataset(x):
    for i in range(len(x[0])):
        x[:, i] /= max(x[:, i])
