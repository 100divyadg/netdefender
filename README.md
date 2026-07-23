# NetDefender

**Machine learning vs deep learning for network intrusion detection.**

NetDefender trains three models on the NSL-KDD dataset and measures whether deep
learning actually improves on a classical baseline for detecting network attacks.
A Random Forest serves as the benchmark; an MLP and a CNN-LSTM hybrid are the
challengers. Everything is wrapped in a Flask dashboard that classifies traffic
records in real time.

Traffic is sorted into five classes: **Normal**, **DoS**, **Probe**, **R2L**
(remote to local) and **U2R** (user to root).

---

## Why this is harder than the accuracy number suggests

NSL-KDD is deliberately awkward, and that is the point of the project.

The `KDDTest+` split contains attack types that never appear in training, so it
measures generalisation to unseen attacks rather than memorisation. The classes
are also severely imbalanced: U2R has roughly fifty training examples against
tens of thousands of DoS records. A model that predicts "Normal" for everything
scores well on raw accuracy while catching no intrusions at all.

For that reason every result here reports **macro averaged precision, recall and
F1** alongside accuracy, and calls out U2R and R2L recall separately. Those two
columns are where the models genuinely differ.

---

## Results

Random Forest baseline, trained with SMOTE balancing and evaluated on `KDDTest+`:

| Metric | Score |
|---|---|
| Accuracy | 0.754 |
| Precision (macro) | 0.793 |
| Recall (macro) | 0.509 |
| F1 (macro) | 0.534 |
| AUC-ROC (ovr) | 0.923 |

Per class:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| DoS | 0.962 | 0.783 | 0.864 | 7460 |
| Normal | 0.653 | 0.973 | 0.782 | 9711 |
| Probe | 0.836 | 0.608 | 0.704 | 2421 |
| R2L | 0.974 | 0.078 | 0.144 | 2885 |
| U2R | 0.539 | 0.105 | 0.175 | 67 |

The pattern is the well documented NSL-KDD result: DoS and Probe are handled
well, while R2L and U2R collapse. R2L precision of 0.974 against recall of 0.078
says the model is correct almost every time it flags an R2L attack, but it flags
almost none of them. Closing that gap is what the deep models are for.

Run `python -m src.train` to reproduce and to fill in the MLP and CNN-LSTM rows;
results land in `results/comparison.csv` and are rendered on the dashboard's
model results page.

---

## Quickstart

```bash
git clone https://github.com/<your-username>/netdefender.git
cd netdefender

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data.download      # fetches NSL-KDD into data/raw
python -m src.train              # trains all three models
python app/app.py                # dashboard at http://localhost:5000
```

Only want the fast baseline? `python -m src.train --models rf` runs in under a
minute and does not require TensorFlow.

---

## Usage

Training options:

```bash
python -m src.train                     # all three models, SMOTE enabled
python -m src.train --models rf mlp     # a subset
python -m src.train --no-smote          # ablation: no class balancing
python -m src.train --pca               # ablation: PCA to 30 components
```

Programmatic prediction:

```python
from src.predict import Detector

detector = Detector("random_forest")     # or "mlp", "cnn_lstm"

result = detector.predict({
    "protocol_type": "tcp",
    "service": "private",
    "flag": "S0",
    "src_bytes": 0,
    "count": 511,
    "serror_rate": 1.0,
})

print(result["prediction"])   # DoS
print(result["confidence"])   # 0.99
```

Scoring a whole capture file:

```python
import pandas as pd
scored = detector.predict_batch(pd.read_csv("capture.csv"))
print(scored["prediction"].value_counts())
```

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Monitoring dashboard |
| GET | `/results` | Model comparison page |
| GET | `/api/health` | Service status and which models are trained |
| POST | `/api/predict` | Classify one record (JSON body) |
| POST | `/api/predict/batch` | Classify an uploaded CSV |
| GET | `/api/alerts` | Attacks flagged this session |
| GET | `/api/metrics` | Stored evaluation metrics |

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"protocol_type":"tcp","service":"private","flag":"S0","count":511,"serror_rate":1}'
```

---

## How it works

**Preprocessing** (`src/data/preprocess.py`) collapses the 39 granular NSL-KDD
labels into the five headline classes, one hot encodes `protocol_type`,
`service` and `flag`, and min max scales the numeric columns. This expands 41
raw features to 121. SMOTE then oversamples the minority classes so U2R is not
drowned out. The fitted transformers are saved to `models/` so the web app
applies exactly the same transformation it was trained on.

**Random Forest** (`src/models/random_forest.py`) is the benchmark: 200 trees
with balanced subsample weighting. It trains in seconds and yields a feature
importance ranking, which is what makes its output legible to an analyst.

**MLP** (`src/models/mlp.py`) is a 256-128-64 fully connected network with batch
normalisation and dropout. It isolates how much of the deep learning gain comes
from depth alone.

**CNN-LSTM** (`src/models/cnn_lstm.py`) treats the feature vector as a sequence.
Conv1D blocks pick up local groups of correlated features, such as the cluster
of `dst_host_*` statistics that together characterise a port sweep, and an LSTM
layer then models longer range dependencies across the vector.

Both deep models use early stopping on validation loss with learning rate
reduction on plateau.

---

## Project layout

```
netdefender/
├── src/
│   ├── config.py              # all paths, hyperparameters, attack taxonomy
│   ├── train.py               # training and comparison entry point
│   ├── predict.py             # inference service used by the web app
│   ├── data/
│   │   ├── download.py        # fetches NSL-KDD
│   │   └── preprocess.py      # encoding, scaling, SMOTE, PCA
│   ├── models/
│   │   ├── random_forest.py
│   │   ├── mlp.py
│   │   └── cnn_lstm.py
│   └── utils/evaluate.py      # metrics, confusion matrices, plots
├── app/
│   ├── app.py                 # Flask backend
│   ├── templates/             # dashboard and results pages
│   └── static/                # stylesheet and dashboard JS
├── tests/test_pipeline.py
├── results/                   # metrics and figures land here
└── requirements.txt
```

---

## A note on the label ordering trap

`CLASS_NAMES` in `src/config.py` **must stay sorted**. scikit-learn's
`LabelEncoder` assigns integer codes alphabetically, and the prediction path
converts predicted integers back to names by indexing into `CLASS_NAMES`. If the
two orderings disagree, the system returns confident but completely wrong
labels: a SYN flood comes back as `Normal` with 100% confidence, and the metrics
look plausible because the support counts still add up.

The pipeline raises immediately if the two ever diverge, and
`tests/test_pipeline.py::TestLabelOrdering` guards it.

---

## Development

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
ruff check src/ app/ tests/
```

---

## Dataset

NSL-KDD, from the Canadian Institute for Cybersecurity at the University of New
Brunswick: https://www.unb.ca/cic/datasets/nsl.html

It is not committed to the repository. `python -m src.data.download` fetches it.

---

## Limitations

This is a research and coursework project, not a production IDS. It classifies
pre extracted connection records rather than live packets, so deploying it for
real monitoring would need a feature extraction layer (a Zeek or CICFlowMeter
style pipeline) in front of it. NSL-KDD is also a 1999-era traffic profile;
performance on modern encrypted traffic would need re-validation on a current
dataset such as CIC-IDS2017 or UNSW-NB15.

---

## Roadmap

- Live packet capture module feeding the detector from a network interface
- Transfer learning for zero day attack types
- Adversarial robustness testing against evasion attacks
- Federated learning for edge deployment on routers and IoT hubs
- Autoencoder and transformer embeddings for richer feature representations

---

## References

1. Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set. *IEEE CISDA*, 1-6.
2. Moustafa, N., & Slay, J. (2016). The UNSW-NB15 dataset. *MilCIS*, 1-6.
3. Aljawarneh, S., Aldwairi, M., & Yassein, M. B. (2018). Anomaly-based intrusion detection through feature selection analysis. *Journal of Computational Science*, 25, 152-160.
4. Vinayakumar, R., et al. (2019). A deep learning approach for intelligent intrusion detection system. *IEEE Access*, 7, 41525-41550.
5. Javaid, A., Niyaz, Q., Sun, W., & Alam, M. (2016). A deep learning approach for network intrusion detection system. *EAI BICT*, 21-26.
6. Kwon, D., & Ahn, S. (2020). CNN and LSTM-based deep learning model for intrusion detection system. *Applied Sciences*, 10(15), 5278.
7. Wang, Y., & Sheng, V. S. (2022). A survey on deep learning-based intrusion detection systems. *Neural Computing and Applications*, 34(5), 3129-3152.

---

## License

MIT. See [LICENSE](LICENSE).
