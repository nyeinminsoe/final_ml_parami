# Customer Classification Using K-Means Clustering

An end-to-end unsupervised machine learning project that segments Portuguese bank marketing customers using **K-Means Clustering** with **Principal Component Analysis (PCA)**, deployed as an interactive **Streamlit** web application.

* **Live Demo:** [Streamlit App]([https://nms25kmeansclfapp.streamlit.app/]))
* **Course:** Introduction to Machine Learning — Parami University
* **Author:** Nyein Min Soe

---

## Project Overview

Direct marketing campaigns often rely on broad outreach, resulting in low conversion rates and wasted effort. This project groups bank customers into distinct behavioral and demographic segments so marketing teams can tailor outreach strategies to each profile.

---

## Dataset Overview

* **Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank.csv`, semicolon-separated)
* **Shape:** `4,521` rows × `15` features (after dropping the target column `y` and call `duration` to prevent data leakage)
* **Data Quality:** `0` missing values
* **Feature Breakdown:**
  * **9 Numeric / Binary Columns:** `age`, `balance`, `day`, `campaign`, `pdays`, `previous`, plus binary-encoded `default`, `housing`, and `loan`
  * **6 Categorical Columns:** `job`, `marital`, `education`, `contact`, `month`, `poutcome`

---

## Methodology & Model Evaluation

### 1. Data Preprocessing & Exploratory Analysis
* **Binary Encoding:** Mapped `'yes'` / `'no'` values in `default`, `housing`, and `loan` to `1` / `0` to reduce dimensionality before one-hot encoding.
* **Correlation & Distribution Checks:** Confirmed low multicollinearity across numeric features, retaining all 15 columns. Identified right-skewed distributions across variables such as `balance`, `campaign`, `pdays`, and `previous`.

### 2. Dimensionality Reduction & Clustering Evaluation ($k = 1 \dots 20$)
Two preprocessing pipelines were evaluated using **Within-Cluster Sum of Squares (WCSS)** and **Silhouette Score**:

| Metric | Without PCA | With PCA (`n_components=2`) | Impact |
| :--- | :--- | :--- | :--- |
| **WCSS Range** | `200,000` – `120,000` | `30,000` – near `0` | Tighter, more compact clusters with a clear elbow at **$k = 3$** |
| **Silhouette Score** | `~0.04` – `0.11` | Up to **`~0.55`** (at $k = 3$) | Substantial improvement in cluster separation |

### 3. Finalized Scikit-Learn Pipeline
```text
Categorical Cols ──► OneHotEncoder ──┐
                                     ├──► StandardScaler ──► PCA (n_components=2) ──► KMeans (n_clusters=3)
Numeric Cols ──────► Passthrough ────┘
```

---

## Streamlit Dashboard Features

* **Cluster Overview Metrics:** Displays customer count and percentage share of the dataset for each segment (`Cluster 0`, `Cluster 1`, `Cluster 2`).
* **Side-by-Side Cluster Profiling:** Summarizes the selected cluster's most frequent categorical attributes (**mode**) alongside average numerical attributes (**mean**).
* **Interactive Scatter Plot:** Lets users dynamically choose X and Y features (defaulting to `age` vs. `balance`) to compare the selected cluster against the rest of the customer base.

---

## Repository Structure

```text
├── bank.csv                  # Semicolon-separated Bank Marketing dataset
├── bank_kmeans.ipynb         # EDA, PCA comparison, Elbow/Silhouette evaluation, and model export
├── bank_kmeans.pkl           # Serialized scikit-learn pipeline (Preprocessing + PCA + K-Means)
├── bank_kmeans_2.py          # Streamlit dashboard application
└── README.md                 # Project documentation
```

---

## Getting Started (Local Setup)

### Prerequisites
* Python 3.8 or higher

### Installation & Running
1. **Clone the repository:**
   ```bash
   git clone https://github.com/nyeinminsoe/final_ml_parami.git
   cd final_ml_parami
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run bank_kmeans.py
   ```
   Open `http://localhost:8501` in your browser, or use the [live Streamlit deployment](https://finalmlparami-48zjd4w8wuf4i8vn7z9pl2.streamlit.app/).

4. *(Optional)* **Re-run the notebook:**
   Open `bank_kmeans.ipynb` in Jupyter Notebook or VS Code to inspect the exploratory plots and regenerate `bank_kmeans.pkl`.

---

## Tech Stack
* **Data Processing:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `PCA`, `KMeans`), `joblib`
* **Visualization & App:** `streamlit`, `matplotlib`, `seaborn`
