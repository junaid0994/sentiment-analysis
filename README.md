# AI-Powered Sentiment Analysis System

An NLP + machine learning project that classifies customer reviews as **Positive**, **Negative** or **Neutral**. The trained model is served through a small Streamlit app that shows the prediction, a confidence score and the probability of each class.

## Screenshots

| Positive | Negative | Neutral |
|---|---|---|
| ![Positive](screenshot_positive.png) | ![Negative](screenshot_negative.png) | ![Neutral](screenshot_neutral.png) |

## Tech stack

Python, pandas, scikit-learn, matplotlib, seaborn, joblib, Streamlit

## Dataset

`reviews.csv` has 10,000 customer reviews with these columns: `review_id`, `review_text`, `sentiment` (target), `category`, `source`.

**Limitations**
- Only **300 of the 10,000 review texts are unique** (100 per class). Duplicate texts were removed before training, otherwise the same review lands in both train and test.
- Even after that, the reviews are built from about 72 repeating sentences, so some test sentences also appear in training.
- Because of this, the model scores almost 100%. That shows the dataset is easy, not that the model will be perfect on real reviews.
- `category` and `source` are spread evenly across sentiments, so they are not used as model input.

## Workflow

1. **Data cleaning:** checked missing values, empty reviews, duplicates and conflicting labels. Removed duplicate texts (10,000 to 300 rows).
2. **EDA:** class balance, review length per class, category and source plots, common words per class.
3. **Text preprocessing:** lowercasing and punctuation are handled by the vectorizer. A custom stopword list keeps "not", "no" and "never", because removing them can flip the meaning of a review.
4. **Feature extraction:** Bag of Words and TF-IDF compared.
5. **Model training:** Logistic Regression, Multinomial Naive Bayes and Linear SVM, compared with 5-fold cross-validation.
6. **Evaluation:** accuracy, precision, recall, F1, confusion matrix and one-vs-rest ROC AUC.
7. **Tuning:** grid search (5-fold CV, macro F1) over `C` and `ngram_range`.
8. **Pipeline:** TF-IDF and Logistic Regression saved together as one file with joblib.
9. **App:** Streamlit interface for predicting new reviews.

## Final model

TF-IDF (unigrams and bigrams) with Logistic Regression, `C=1`. Linear SVM scored about the same, but it gives no probabilities, and the app needs a confidence score.

## Results

| Check | Result |
|---|---|
| 5-fold CV macro F1 (training set, best model) | 1.00 |
| Test accuracy (60 reviews) | 1.00 |
| Precision / recall / F1 (each class) | 1.00 |
| ROC AUC (one-vs-rest) | 1.00 |

These perfect scores come from the dataset limits above. See `EVALUATION_REPORT.md` for the full model comparison.

## Project structure

```
reviews.csv
sentiment_analysis.ipynb
sentiment_pipeline.pkl
app.py
requirements.txt
README.md
EVALUATION_REPORT.md
screenshot_positive.png
screenshot_negative.png
screenshot_neutral.png
```

## How to run

1. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
2. Open `sentiment_analysis.ipynb` and run all cells. The last cells save the model as `sentiment_pipeline.pkl`.
3. Start the app from the same folder:
   ```
   python -m streamlit run app.py
   ```

## Try it

- "Excellent quality, I would happily buy it again" is predicted Positive.
- "Customer support was slow and did not solve my issue" is predicted Negative.
- "The box contains a charger and a user manual" is predicted Neutral.

The model was trained on short, simple reviews, so it may struggle with reviews that mix positive and negative opinions.
