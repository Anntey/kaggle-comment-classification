# Kaggle: Toxic Comment Classification Challenge ([link](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview))

Data: 153 165 Wikipedia comments

Task: predict the probability for each of the six possible types of comment toxicity

Evaluation: mean column-wise ROC AUC (i.e. the score is the average of the individual AUCs of each predicted column)

Solution: bi-directional LSTM-RNN using GloVe word embeddings

Success: 0.9784 AUC
&nbsp;

> Before you start throwing accusations and warnings at me, lets review the edit itself-making ad hominem attacks isn't going to strengthen your argument, it will merely make it look like you are abusin...

> Oh, and the girl above started her arguments with me. She stuck her nose where it doesn't belong. I believe the argument was between me and Yvesnimmo. But like I said, the situation was settled and I ...

> Don't mean to bother you I see that you're writing something regarding removing anything posted here and if you do oh well but if not and you can acctually discuss this with me then even better. I'...

> I was able to post the above list so quickly because I already had it in a text file in my hard drive I've been meaning to get around to updating the sound list for some time now. As far as generati...
