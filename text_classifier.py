import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import movie_reviews


'''
Based off the standard code example given on nltk.org, but 
instead of using a large list of generally common words for the feature extractor, 
I decided to supply words usually
associated with positive or negative reviews in order to 
help give a simpler program that runs faster, takes less memory,
and is more specifically targeted to the movie reviews

Additionally, the testing to training split was changed to 
80 - 20 

'''
positive_list = ["great", "excellent", "marvellous","superb","good","exemplary",
"outstanding","skillful","notable","exemplary"]

negative_list = ["atrocious","horrendous","horrid","awful","disastrous","bad","terrible",
"boring","mediocre","unoriginal","copycat","predicatable","cliche"]

documents = [(list(movie_reviews.words(fileid)), category)
			for category in movie_reviews.categories()
			for fileid in movie_reviews.fileids(category)]

total_list = ["atrocious","horrendous","horrid","awful","disastrous","bad","terrible",
"boring","mediocre","unoriginal","copycat","predicatable","cliche"]

word_features = total_list

def document_features(document): 
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
    
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[80:], featuresets[:20]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print nltk.classify.accuracy(classifier, test_set)
classifier.show_most_informative_features(10)