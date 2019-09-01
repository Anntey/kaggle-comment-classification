# Kaggle: Toxic Comment Classification Challenge ([link](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview))

Data: 153 165 Wikipedia comments

Task: predict the probability for each of the six possible types of comment toxicity

Evaluation: mean column-wise ROC AUC (i.e. the score is the average of the individual AUCs of each predicted column)

Solution: bi-directional LSTM-RNN using GloVe word embeddings

Success: 0.9784 AUC

> 	Before you start throwing accusations and warnings at me, lets review the edit itself-making ad hominem attacks isn't going to strengthen your argument, it will merely make it look like you are abusin...
