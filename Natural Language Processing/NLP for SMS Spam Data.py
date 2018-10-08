#nltk.download_shell()

messages = [line.rstrip() for line in open('SMSSpamCollection')]
print(len(messages))

for mess_no, message in enumerate(messages[:1]): #Adding numbers and giving a space btween label and message
    print(mess_no, message)
    print('\n')

messages[0]

import pandas as pd

messages = pd.read_csv('SMSSpamCollection', sep='\t', names =['label', 'message']) #Reading with pandas and adding label and messages as header

print messages
messages.describe() #Data Exploring

messages.groupby('label').describe() #Knowing the data with label


messages['length'] = messages['message'].apply(len) #With Length


messages.head()



import matplotlib.pyplot as plt
import seaborn as sns


get_ipython().magic(u'matplotlib inline')

messages['length'].plot.hist(bins=50)  #Plotting the whole messages file with length

messages['length'].describe() #With feature engineering exploring the data with length


#messages[messages['length'] == 910]['message'].iloc[0] 


messages.hist(column='length', by='label', bins=50, figsize=(12,4)) #plotting as histogram with length of the each ham and spam label 


import string

string.punctuation


from nltk.corpus import stopwords #importing sport words

stopwords.words('english')



def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation] #removing the punctuations
    nopunc = ''.join(nopunc) #removing the stop words
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')] #return list of clean text words

messages['message'].head(5).apply(text_process) #List of clean text words



from sklearn.feature_extraction.text import CountVectorizer #Import count vectorizer


bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message']) 


print(len(bow_transformer.vocabulary_))


mess4 = messages['message'][3] 


print mess4


bow4 = bow_transformer.transform([mess4])

print bow4

print bow4.shape


bow_transformer.get_feature_names()[4629]


message_bow = bow_transformer.transform(messages['message']) #transforming the messages as vectors to form as sparse matrix


print 'shape of the matrix:', message_bow.shape #Shape of sparse matrix


message_bow.nnz #checkin how many non zero elements are there



#Sparsity: comparing non zero vs total number of messages 



sparsity = (100.0 * message_bow.nnz / (message_bow.shape[0] * message_bow.shape[1]))
print('sparsity: {}'.format((sparsity)))




from sklearn.feature_extraction.text import TfidfTransformer #calculating the Term frequency and Inverse document frequency
        
tfidf_transformer = TfidfTransformer().fit(message_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

messages_tfidf=tfidf_transformer.transform(message_bow) #calculating TF-IDF for bag of words

#Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])


#Detection filter using Naive Bayes for Label 2
print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', messages.label[2])

#Model Evaluation of all
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)  


#Classification report
from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))


#Train and Test Split
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))



from sklearn.pipeline import Pipeline


#Decision Tree Classifier
from sklearn import tree


pipeline =Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', tree.DecisionTreeClassifier())
])


#creating a pipeline object
pipeline.fit(msg_train,label_train)


predictions = pipeline.predict(msg_test)

#Classification report for Decision Tree Classfier
from sklearn.metrics import classification_report

print(classification_report(label_test,predictions))


#Classification report for Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
pipeline =Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', RandomForestClassifier())
])
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)

from sklearn.metrics import classification_report
print(classification_report(label_test,predictions))


