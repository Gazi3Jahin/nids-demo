# demo_dataset.py
import pandas as pd
import numpy as np

def generate_demo_dataset(n_samples=2000):
    np.random.seed(42)

    # CICIDS2017-inspired numeric features
    data = {
        "FlowDuration": np.random.randint(1, int(1e6), n_samples),
        "TotalFwdPackets": np.random.randint(1, 5000, n_samples),
        "TotalBackwardPackets": np.random.randint(1, 5000, n_samples),
        "FwdPacketLengthMean": np.random.rand(n_samples) * 500,
        "BwdPacketLengthMean": np.random.rand(n_samples) * 500,
        "FlowBytesPerSec": np.random.rand(n_samples) * 1e6,
        "FlowPacketsPerSec": np.random.rand(n_samples) * 1000,
    }

    X = pd.DataFrame(data)

    # Labels: 0 = Normal, 1 = Attack (70% normal, 30% attack)
    y = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    df = X.copy()
    df["Label"] = y
    return df

if __name__ == "__main__":
    df = generate_demo_dataset()
    df.to_csv("demo_cicids2017.csv", index=False)
    print("✅ Demo dataset saved as demo_cicids2017.csv (rows: {})".format(len(df)))
