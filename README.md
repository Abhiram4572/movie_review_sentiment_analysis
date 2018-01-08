# movie_review_sentiment_analysis
the code classifies sentiments of movie reviews into positive and negative.


code.py - this has the code for classification(80% train data, 20 % cross validation data)
Cv.tsv  - cross validation set
Results.csv - final output file, reviews(texts) are removed, only id and predicted sentiment.



Input files given by you are not included.


Abstract:- 
Sentiment analysis is a challenging subject in machine learning. People express their emotions in language that is often obscured by sarcasm, ambiguity, and plays on words, all of which could be very misleading for both humans and computers. Here I present a sentiment analyzer, with Random Forest Classifier trained on IMDB movie data, and accuracy is measured by varying various tuning parameters. 

Literature survey: -
I have looked at various resources for solving this problem, in [1] I got the order in which code should be done. Earlier I have read data from csv (Comma separated values) files using read_csv function of pandas library, now I learnt from its documentation that it can also be used to read data from tsv(tab separated files) files, by setting “delimiter” variable to “\t”. In [2], I read about main tuning parameters of random forest classification model in scikit. [3], various classifiers i.e., Nearest Neighbours, Linear SVM, RBF SVM, Gaussian Process, Decision Tree, Random Forest, Neural Net, Adaboost, Naive Bayes, QDA, are compared and results are plotted along with accuracy.  Various blogs of [4] helped me revise my basics and get an quick recap of python snippets. I refered these documentations for solving the code, scikit documentation, pandas documentation, BeautifulSoup documentation.

Data Characterization: -
Train Data file shared has three columns, id, review and sentiment. Each movie has 30 reviews. I used CountVectorizer from scikit, in training phase fit transform is done on training data, which first creates vocabulary , where each word in vocabulary is a feature. Next it converts the existing data into matrix of features. 
From 
id	review	sentiment
		
To(after transform)

id	word1	word2	so..on
			
			


Feature Engineering: - 
Two different cases where features, i.e., vocab is created by fit_transform function.
Case 1 - 2500 features.
Case 2 - 5000 features.


Model Selection: - 
Model selected is Random Forest, from literature, Random forests accuracy is more. But other open blogs like quora, stack exchange, comments read SVM performs better, but I selected random forest [1], and I’m having experience on it.

Parameter tuning: - 
1.	Features
2.	N_estimators - No. of trees in forest
3.	Criterion - gini or entropy
4.	Min_samples_split - minimum number of working set size at node required to split
Cases: - 
1)	With 50 n_estimators
2)	With 100 n_estimators
3)	With gini as criterion
4)	With entropy as criterion
5)	With 10 min_samples_split
6)	With 2 min_samples_split


Metric used is f1_score(y_true, y_pred) from scikit. 
Y_true =  true class
Y_pred = predicted class
Conclusion and Inferences: - 
1)	Accuracy difference is clearly seen in 2 cases( table 1 and 2), in table 1, model is trained on 100% data and tested on last 20% data. Whereas in table 2, model is trained on 80% data - training set and tested on 20% data -  cross validation set. 
2)	N_estimators - increase in no. of trees in random forest, increased the accuracy.
3)	Increase in number of features has also increased accuracy.
4)	Entropy is better over gini (Impurity).
5)	Min_samples_split - decrease of it, has brought no significant changes.

References: - 
1.	Bag of words tutorial  - https://www.kaggle.com/c/word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words
2.	https://medium.com/@Currie32/predicting-movie-review-sentiment-with-tensorflow-and-tensorboard-53bf16af0acf
3.	Comparison of Classifiers- http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
4.	Decision Tree Classifier - https://medium.com/machine-learning-101/chapter-3-decision-trees-theory-e7398adac567
5.	Random Forest Classifier - https://medium.com/machine-learning-101/chapter-5-random-forest-classifier-56dc7425c3e1



