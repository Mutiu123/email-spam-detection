# import libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle as pkl

#read data file
data = pd.read_csv('spam.csv')
#print(data.head(4))
#print(data.shape)

#clean the data
data.drop_duplicates(inplace=True)
print('')
#print(data.shape)
#check for null value data
#print(data.isnull().sum())

#change category name (ham to not Spam)
data['Category'] = data['Category'].replace(['ham', 'Spam'],['Not Spam', 'Spam'])
#print(data.head(4))

#seperate input and output dataset
cate = data['Category']
messg = data['Message']

#slipt data into training ada testing dataset
(messg_train, messg_test,cate_train,cate_test) = train_test_split(messg, cate, test_size=0.25)

#convert category data to decimal data
cv = CountVectorizer(stop_words='english') 
input_features = cv.fit_transform(messg_train)

#create model
model = MultinomialNB()
model.fit(input_features, cate_train)

#Test the model
test_features = cv.transform(messg_test)
#print(model.score(test_features, cate_test))
print('')

#predict ddata
def predict(message):
    incoming_message = cv.transform([message]).toarray()
    result = model.predict(incoming_message)
    return result[0]

#print(predict('Congratulation, you have won a lotery'))
pkl.dump(model, open('model/ESPD.pkl','wb'))