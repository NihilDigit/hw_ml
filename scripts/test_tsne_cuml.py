import time
import pandas as pd
from cuml.manifold import TSNE

FEATURES = [
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean",
    "Fwd Pkt Len Std",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean",
    "Bwd Pkt Len Std",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "SYN Flag Cnt",
    "ACK Flag Cnt",
    "PSH Flag Cnt",
    "FIN Flag Cnt",
    "RST Flag Cnt",
    "Active Mean",
    "Idle Mean",
]

path = "data/processed/ids2018_subset_10k.csv"
print("Loading", path)
df = pd.read_csv(path, usecols=FEATURES)
X = df[FEATURES].values
print("Shape", X.shape)

start = time.time()
model = TSNE(
    n_components=20,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    n_iter=1000,
    random_state=42,
)
Y = model.fit_transform(X)
print("Done. Output shape", Y.shape)
print("Elapsed", time.time() - start)
