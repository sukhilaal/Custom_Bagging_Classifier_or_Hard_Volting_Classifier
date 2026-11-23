# Heterogeneous Ensemble Learning: Voting Classifier

An implementation and comparative evaluation of a **Heterogeneous Ensemble Model** using Scikit-Learn's `VotingClassifier`. This Jupyter Notebook demonstrates how diverse machine learning algorithms can be combined into a single meta-classifier and compares individual base model accuracies against the aggregated ensemble.

---

## 📌 Project Overview

Unlike homogeneous ensembles (such as Random Forests, which combine many instances of the same algorithm), a **heterogeneous ensemble** combines predictions from fundamentally different model families. By leveraging algorithms with distinct decision boundaries—linear, tree-based, and margin-based—the ensemble balances individual model biases and variances.

### Key Concepts Covered
* **Heterogeneous Ensembling:** Combining distinct classifiers (`LogisticRegression`, `DecisionTreeClassifier`, and `SVC`).
* **Soft Voting (`voting='soft'`):** Predicting class labels based on the argmax of the sums of predicted probabilities across all base estimators (requires `probability=True` for `SVC`).
* **Model Benchmarking:** Evaluating each standalone estimator on the same test split to analyze how the ensemble performs relative to its strongest and weakest learners.

---

## 📊 Dataset Details

The notebook generates a synthetic binary classification dataset using `sklearn.datasets.make_classification`:

* **Total Samples:** $1,000$
* **Number of Features:** $20$
* **Target Classes:** $2$ (Binary Classification)
* **Train / Test Split:** $80\%$ Training ($800 \times 20$) / $20\%$ Testing ($200 \times 20$) with `random_state=1`

---

## 🧠 Model Architecture

The `VotingClassifier` aggregates predicted probabilities (`voting='soft'`) from three diverse base estimators:

```text
Root: Heterogeneous Voting Classifier (Soft Voting)
  ├── Base Model 1: Logistic Regression (Linear probabilistic model)
  ├── Base Model 2: Decision Tree Classifier (Non-linear rule-based model)
  └── Base Model 3: Support Vector Classifier (Linear kernel with probability=True)
```

---

## 📈 Experimental Results

All models were trained on `X_train` and evaluated on `X_test` using classification accuracy:

| Model | Role | Accuracy |
| :--- | :--- | :---: |
| **Support Vector Classifier (`SVC`)** | Base Estimator | **86.50%** (`0.8650`) |
| **Logistic Regression** | Base Estimator | **86.00%** (`0.8600`) |
| **Voting Classifier (Soft)** | **Ensemble Model** | **84.00%** (`0.8400`) |
| **Decision Tree Classifier** | Base Estimator | **76.50%** (`0.7650`) |

### Key Takeaways
* **Variance Reduction:** The Voting Classifier (`84.00%`) significantly outperforms the weakest base learner (`Decision Tree` at `76.50%`), protecting against single-model overfitting.
* **Impact of Equal Weighting:** Because all three estimators carry equal weight by default, the lower accuracy and uncalibrated leaf probabilities of the unpruned Decision Tree pull the ensemble average slightly below the two strong linear classifiers (`SVC` and `Logistic Regression`).

---

## 🛠️ Tech Stack & Dependencies

* **Python 3.x**
* **Scikit-Learn** – Synthetic dataset generation, train-test splitting, base classifiers, `VotingClassifier`, and accuracy evaluation.
* **Jupyter Notebook / Google Colab** – Interactive execution environment.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies
```bash
pip install scikit-learn jupyter
```

### 3. Run the Notebook
Launch Jupyter Notebook or open the `.ipynb` file in Google Colab:
```bash
jupyter notebook your_notebook_name.ipynb
```

---

## 🔮 Future Enhancements

* **Hard vs. Soft Voting Comparison:** Benchmark `voting='hard'` (majority rule voting) directly against `voting='soft'`.
* **Weighted Voting:** Pass custom `weights` (e.g., `weights=[2, 1, 2]`) to `VotingClassifier` so stronger base models have greater influence over the final prediction.
* **Expand Estimator Diversity:** Integrate `GaussianNB` and `RandomForestClassifier` into the ensemble pipeline and evaluate stability using $k$-fold cross-validation (`cross_val_score`).
