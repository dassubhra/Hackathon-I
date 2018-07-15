###########################################################################
''' Import the required libraries '''

#Using the below given methods is one way to train your classifier
#Define your classifier in this file
###########################################################################
import Record_audio
import glob
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def completeTheClassification():
    for filename in glob.iglob("/Users/madhukaruturi/Desktop/Madhu/AI-ML/AIML-Labs/Hackathon I/speech-data/*.wav"):
        featureDF.append(Record_audio.get_features(filename, sr=8000, n_mfcc=30, n_mels=128, frames = 15))
        labelDF.append(os.path.basename(filename).split('_')[0])
    
def useKNN():
    finalModel = '/Users/madhukaruturi/Desktop/Madhu/AI-ML/AIML-Labs/Hackathon I/model.sav'
    train, test, train_labels, test_labels = train_test_split(featureDF,labelDF,test_size=0.2,random_state=11)
    knn = KNeighborsClassifier(n_neighbors=16,metric='euclidean')
    knn.fit(train, train_labels)
    score = knn.score(train, train_labels)
    pred = knn.predict(test)
    pickle.dump(knn,open(finalModel,'wb'))
    print(score,pred,accuracy_score(test_labels, pred))
   
def test(fileName):
    deep_features =[]
    deep_features.append(Record_audio.get_features(fileName, sr=8000, n_mfcc=30, n_mels=128, frames = 15))
    loadedModel = pickle.load(open("/Users/madhukaruturi/Desktop/Madhu/AI-ML/AIML-Labs/Hackathon I/model.sav",'rb'))
    print(loadedModel.predict(deep_features))
    
#completeTheClassification()    
#useKNN()
test("speech-data/1_userinput_1.wav")