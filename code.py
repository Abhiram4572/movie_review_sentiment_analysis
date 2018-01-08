import pandas as pd
import re
import nltk
import numpy as np
from bs4 import BeautifulSoup 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from nltk.corpus import stopwords

def review_clean(review):
	text_review = BeautifulSoup(review, "lxml").get_text(); #Cleaning the review from all <br> tags
	Clean_from_Punc = re.sub("^[a-zA-Z]", " ", text_review) #replacing all punctuations with " " space
	words = Clean_from_Punc.lower().split()                 #Converting all chars to lowercase and spliting them
	stop = set(stopwords.words("english"))                  #stop contains all stop words like I, we etc.
	final_words = [w for w in words if not w in stop]      #removing all stop words from review
	return (" ".join(final_words))                          #return array of all words
	
train_data = pd.read_csv("labeledTrainData.tsv", delimiter = "\t")    #reading the train data
print(train_data.shape)
test_data = pd.read_csv("testData.tsv", delimiter = "\t")             #reading the test data
print(test_data.shape)	



#data division


cv = []
cv2 = []

for i in range(20000,25000):
	cv.append(train_data["review"][i])
	cv2.append(train_data["sentiment"][i])
	

output = pd.DataFrame(data = {"review":cv, "sentiment":cv2})
output.to_csv("Cv.tsv", index= False, sep="\t")

cv_data = pd.read_csv("Cv.tsv", delimiter = "\t")  


t2 = []
for i in range(0,20000):
	t2.append(train_data["sentiment"][i])

#train



num = len(train_data["review"])

clean_reviews_train = []

for i in range(0,20000):                                              #cleaning all reviews
	clean_review = review_clean(train_data["review"][i])
	clean_reviews_train.append(clean_review)

vect = CountVectorizer(max_features = 2500)                           #Initialize CountVectorizer
train_features = vect.fit_transform(clean_reviews_train)              #fit reviews

train_features = train_features.toarray()
print(train_features.shape)

forest = RandomForestClassifier(n_estimators = 100, criterion = "entropy", min_samples_split = 2)                    #Initializing RF classifier with tuning params


forest = forest.fit(train_features, t2)




#cv


clean_review_crossvalid = []    
orig = cv_data["sentiment"]                                        #cross validation check, accuracy check

for i in range(0,5000):                                            #cleaning cv set
	clean_review_cv = review_clean(cv_data["review"][i])
	clean_review_crossvalid.append(clean_review_cv)

cv_features = vect.transform(clean_review_crossvalid)
cv_features = cv_features.toarray()

cv_result = forest.predict(cv_features)


print(f1_score(orig,cv_result))





#test


clean_review_test = []

for i in range(0,25000):
	clean_review = review_clean(test_data["review"][i])
	clean_review_test.append(clean_review)

test_features = vect.transform(clean_review_test)
test_features = test_features.toarray()

test_result = forest.predict(test_features)

output = pd.DataFrame(data ={"id":test_data["id"], "sentiment":test_result} )
output.to_csv("Result.csv", index = False, sep="\t")	
	
	
	
	
	
