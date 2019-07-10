# Kaggle: Toxic Comment Classification Challenge

Data: 153 165 Wikipedia comments

Task: predict the probability for each of the six possible types of comment toxicity

Evaluation: mean column-wise ROC AUC (i.e. the score is the average of the individual AUCs of each predicted column)

Solution: Bidirectional LSTM using GloVe word embeddings (Python, Keras)

