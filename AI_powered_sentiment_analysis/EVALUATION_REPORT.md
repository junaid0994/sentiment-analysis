# Evaluation Report

All numbers come from `sentiment_analysis.ipynb` (random_state=42). Training and cross-validation use the 240 training reviews; the 60 test reviews are used once at the end.

## 1. Data used

10,000 rows -> 300 unique reviews after removing duplicate texts (100 per class). Without this step, a random split puts every test review in the training set as well, and the score is meaningless.

## 2. Stopword choice (TF-IDF + Logistic Regression, 5-fold CV accuracy)

| Option | CV accuracy |
|---|---|
| No stopword removal | 0.992 |
| sklearn English list | 0.996 |
| Custom list (keeps not, no, never) | 0.996 |

The scores are almost equal on this dataset. A direct check shows the difference: with sklearn's list the model predicted "I would recommend it" as Negative and "This is not good" as Positive. The custom list got both right. The custom list is used because it is the safer choice for real reviews.

## 3. Features and models (5-fold CV accuracy)

| Model | Bag of Words | TF-IDF |
|---|---|---|
| Logistic Regression | 0.996 | 0.996 |
| Multinomial Naive Bayes | 0.996 | 0.996 |
| Linear SVM | 1.000 | 1.000 |

The differences are too small to separate the options. TF-IDF is used as the standard choice for text.

## 4. Tuning

Grid search with 5-fold CV, scoring macro F1, on `C` (0.1, 1, 10) and `ngram_range` ((1,1), (1,2)).

Best: `C=1`, `ngram_range=(1,2)`, CV macro F1 = 1.00.

## 5. Final test results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 1.00 | 1.00 | 1.00 | 20 |
| Neutral | 1.00 | 1.00 | 1.00 | 20 |
| Positive | 1.00 | 1.00 | 1.00 | 20 |

Accuracy 1.00 on 60 reviews. The confusion matrix is fully diagonal (20/20/20). One-vs-rest ROC AUC = 1.00.

## 6. Why Logistic Regression as the final model

- Its scores match the other models on this data.
- It gives class probabilities, which the app shows as a confidence score. Linear SVM does not give probabilities by default.
- It is simple and easy to explain.

## 7. Limitations

- The 300 unique reviews are built from about 72 repeating sentences, so test sentences also appear in training. The perfect scores show an easy dataset, not a perfect model.
- With only 60 test reviews, one test split is a rough estimate, so cross-validation was used as well.
- Neutral reviews here are plain factual statements, which is easier than real neutral reviews that mix opinions.
- The model has not been tested on real-world reviews.
