# Depression Detection using NLP

This project demonstrates a simple text classification model to detect signs of depression in short sentences using **TF-IDF vectorization** and **Logistic Regression**.

---

## 📊 Dataset
- Source: [Depression Detection Dataset (Hugging Face)](https://huggingface.co/datasets/thePixel42/depression-detection)  
- Original size: 120,000 samples (balanced).  
- For this project, a **20,000 sample subset** was used (**10k per class**) due to:
  - GitHub file size limitations (pickle files >100 MB not allowed).
  - Lack of GPU for training on the full dataset.

---

## ⚙️ Methodology
1. **Preprocessing**
   - Cleaned and balanced dataset (`balanced_20k.csv`).
   - Converted text into numerical features using **TF-IDF Vectorizer**.

2. **Model Training**
   - Classifier: **Logistic Regression**.
   - Saved trained model and vectorizer as `model.pkl` (via joblib).

3. **Deployment Setup**
   - Streamlit app (`app.py`) for interactive predictions.
   - Configured with `Procfile`, `setup.sh`, and `runtime.txt` for Heroku deployment.

---

## 📈 Results (on 4k test set)
| Metric     | Class 0 | Class 1 | Avg. |
|------------|---------|---------|------|
| Precision  | 0.90    | 0.91    | 0.90 |
| Recall     | 0.92    | 0.89    | 0.90 |
| F1-score   | 0.91    | 0.90    | 0.90 |

- **Accuracy:** 90%  
- **Test set size:** ~4,000 sentences  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/AshutoshTiwari0/depression-detection-minor-project.git
   cd depression-detection-minor-project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the streamlit app
   ```bash
   streamlit run app.py

## 📝 Notes

1.Full dataset (120k) yields higher accuracy (~96% with Random Forest), but not included here due to size restrictions.

2.This project demonstrates a practical trade-off between model performance and deployment constraints.

3.Larger datasets generally improve learning, but smaller subsets are often enough for prototyping.

## ✨ Future Improvements

1.Try deep learning models (LSTMs, Transformers) for better semantic understanding.

2.Use contextual embeddings (BERT, RoBERTa) instead of TF-IDF.

3.Collect more real-world test data to evaluate generalization.
