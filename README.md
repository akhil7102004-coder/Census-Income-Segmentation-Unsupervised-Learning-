# 👥 Census Income Segmentation — Unsupervised Learning

> Segmenting 48K+ census records into socioeconomic groups using PCA, K-Means and Agglomerative Clustering, then profiling each group against income.

---

## 📌 Project Overview

This project applies **unsupervised machine learning** to the Adult Census Income dataset to discover natural groups in the population without using the income label for training.

After clustering, the income column is used only to **interpret** the clusters, showing that the groups found by the algorithms differ sharply in income even though income was never given to the model.

---

## 🎯 Objectives

- Clean and preprocess a real-world census dataset
- Explore the data with EDA and dimensionality reduction (PCA)
- Find the number of clusters using the elbow method and silhouette score
- Compare K-Means with Agglomerative (hierarchical) clustering
- Profile each cluster and translate findings into policy suggestions

---

## 🗂️ Dataset

| Property | Value |
| --- | --- |
| File | `adult_dataset.csv` |
| Raw records | 48,842 rows, 15 columns |
| After removing duplicates | 48,826 rows |
| After one-hot encoding | 97 features |
| Target (used only for profiling) | `income` (`<=50K` / `>50K`) |

**Features include:** age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours per week, native country.

---

## 🔄 Workflow

```text
Raw Census Data
      ↓
Data Cleaning (duplicates, '?' values, missing values)
      ↓
Outlier Treatment (IQR clipping)
      ↓
One-Hot Encoding + Standard Scaling
      ↓
EDA + PCA Visualisation
      ↓
K-Means (Elbow + Silhouette) / Agglomerative (Ward + Dendrogram)
      ↓
Cluster Profiling vs Income
      ↓
Interpretation + Policy Suggestions
```

---

## 🧹 Data Preprocessing

- Removed **16 duplicate records**
- Replaced `?` placeholders with missing values
- Imputed missing numeric values with the **median** and categorical values with the **mode**
- Treated outliers in `age`, `fnlwgt` and `hours-per-week` using **IQR clipping**
- Excluded `income` from the features, then applied **one-hot encoding** (`drop_first=True`)
- Standardised all features with **StandardScaler**

---

## 📊 Exploratory Data Analysis

- Histograms of numerical features
- Count plots of the top categories for each categorical feature
- Correlation heatmap of numerical features
- **PCA** to 2 components for visualisation (explains about **6.3%** of the variance, so the plot is a simplified view only)

---

## 🤖 Clustering

### K-Means
- Elbow method and silhouette scores tested for K = 2 to 10
- Final model: **K = 2** (highest silhouette score, 0.0508)
- Cluster sizes: 25,659 (52.55%) and 23,167 (47.45%)

### Agglomerative Clustering (Ward linkage)
- Run on a 10,000-record sample because of computational cost
- Dendrogram drawn on a 1,000-record sample
- Silhouette score: **0.4709**, but the clusters are very imbalanced (966 vs 9,034)

### Comparison on the same sample

| Model | Silhouette Score | Cluster Sizes |
| --- | ---: | --- |
| K-Means (K=2) | 0.0507 | 5,246 / 4,754 |
| Agglomerative (K=2) | 0.4709 | 966 / 9,034 |

> The higher Agglomerative score comes with very uneven clusters, so K-Means was kept as the final model for profiling because it gives balanced, interpretable groups.

---

## 🔍 Cluster Profiles (K-Means)

| Feature | Cluster 0 | Cluster 1 |
| --- | ---: | ---: |
| Average age | 34.24 | 43.29 |
| Average education-num | 9.82 | 10.37 |
| Average hours per week | 39.61 | 42.82 |
| Average capital gain | 350.48 | 1,886.77 |
| Earning `<=50K` | 94.63% | 55.50% |
| Earning `>50K` | 5.37% | 44.50% |
| Common marital status | Never-married | Married-civ-spouse |

- **Cluster 0:** a younger, lower-income group
- **Cluster 1:** an older, higher-income group with much higher capital gains

Income was not used to build the clusters, yet the two groups differ strongly on it.

---

## 💡 Policy Suggestions

- **Upskilling programs** for the lower-income cluster
- **Financial aid and inclusion** such as education assistance, affordable loans, financial literacy and banking access
- **Tax policy** that accounts for the income and capital-gain differences between groups

---

## ⚠️ Limitations

- Silhouette scores for K-Means are low (about 0.05), which means the clusters overlap and are not sharply separated
- The first two principal components explain only about 6.3% of the variance
- The dataset mixes many categorical features, so one-hot encoding creates a high-dimensional space
- Agglomerative Clustering was run on a sample, not the full dataset
- Clusters describe patterns in the data and should not be read as causal

---

## 🛠️ Technologies Used

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, SciPy, Matplotlib, Seaborn
- **Techniques:** PCA, K-Means, Agglomerative Clustering, Silhouette Analysis, Dendrograms
- **Environment:** Jupyter Notebook / Google Colab

---

## 📁 Project Structure

```text
census-income-segmentation/
│
├── UNSUPERVISED_LEARNING_CASE.ipynb
├── adult_dataset.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/akhil7102004-coder/<your-repo-name>.git
cd <your-repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn scipy matplotlib seaborn jupyter
```

## ▶️ Run

```bash
jupyter notebook UNSUPERVISED_LEARNING_CASE.ipynb
```

If you use Google Colab, upload `adult_dataset.csv` and update the file path in the first data-loading cell.

---

## 🔮 Future Improvements

- Try DBSCAN or Gaussian Mixture Models
- Use dimensionality reduction suited to mixed data (e.g. FAMD) or K-Prototypes
- Run Agglomerative Clustering on the full dataset or with a different linkage
- Validate clusters with additional metrics such as Davies–Bouldin and Calinski–Harabasz

---

## 👨‍💻 Author

**Akhil A**
B.Tech Computer Science & Engineering
[LinkedIn](https://linkedin.com/in/akhil-a-48930629b) • [GitHub](https://github.com/akhil7102004-coder)

---

## 📄 License

This project is created for educational, learning, and portfolio purposes.
