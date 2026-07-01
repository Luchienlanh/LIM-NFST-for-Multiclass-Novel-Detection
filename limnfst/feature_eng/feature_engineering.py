import numpy as np
import pandas as pd

from limnfst.preprocess.sampling import stratified_min_per_class_sample


def main():
    rng = np.random.default_rng(0)
    y = np.array(["DDoS"] * 1000 + ["DoS"] * 180 + ["Theft"] * 45)
    X = pd.DataFrame(
        {
            "rate": rng.normal(size=len(y)),
            "bytes": rng.lognormal(size=len(y)),
            "pkts": rng.poisson(lam=5, size=len(y)),
        }
    )

    X_sample, y_sample, idx = stratified_min_per_class_sample(
        X,
        pd.Series(y),
        n_samples=240,
        min_per_class=40,
        random_state=42,
        return_indices=True,
    )

    counts = pd.Series(y_sample).value_counts()
    print("original counts:")
    print(pd.Series(y).value_counts().to_string())
    print()
    print("sample counts:")
    print(counts.to_string())
    print()
    print("X_sample shape:", X_sample.shape)
    print("y_sample shape:", y_sample.shape)
    print("unique sampled rows:", len(np.unique(idx)))

    assert len(X_sample) == 240
    assert len(y_sample) == 240
    assert counts.min() >= 40
    assert len(np.unique(idx)) == len(idx)


if __name__ == "__main__":
    main()
