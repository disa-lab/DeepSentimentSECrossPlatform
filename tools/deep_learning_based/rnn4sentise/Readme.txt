Please Note: 

This code is obtained from the authors of the following paper:
Exploring word embedding techniques to improve sentiment analysis of software engineering texts
Biswas, Eeshita and Vijay-Shanker, K and Pollock, Lori
2019 IEEE/ACM 16th International Conference on Mining Software Repositories (MSR), 68--78, 2019, IEEE

We have modified the code to allow processing of the new dataset (CombinedData).

-----------------------------------------------------------------------------------------------------------

Before running the code, you will need to download all data from the Dataset folder, and place them into the same directory as Sentiment_Classifier.py. 

You will also need to download the word embedding model. We have used the Google News word embeddings for the RNN4SentiSE sentiment classifier as mentioned in the paper. Download the "GoogleNews-vectors-negative300.bin.gz" from the below link, and place them into the same directory as Sentiment_Classifier.py. :
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit


Installing Requirements:

This code is compatible for the following tensor flow and numpy version:
Tensorflow 1.4.0
Numpy 1.16.4

For later versions, check the compatibility between Tensorflow and numpy versions.

Also need to install the gensim library for the Google news word embeddings.



