import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import pandas as pd
import keras.backend as K
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.utils import to_categorical 
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Model
import timeit
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding1D
from tensorflow.keras.optimizers import SGD
import warnings
warnings.filterwarnings('ignore')
from keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, BatchNormalization, Activation, Dropout, Flatten, Dense
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sb
import wordcloud
from wordcloud import WordCloud, STOPWORDS
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk import word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import KFold 

#EDA ANALYSIS

#importing the dataset
df = pd.read_csv("dataset/mbti_1.csv")
print(df)

#Information on the dataset
print(df.info)

#Counting the personality types
count=df.groupby(['type']).count()


#Plotting the personality count
plt.figure(figsize = (12,4))
plt.bar(np.array(count.index), height = count['posts'],)
plt.xlabel('Personality types', size = 14)
plt.ylabel('No. of posts available', size = 14)
plt.title('Total posts for each personality type')
plt.show()

#Finding the length of the posts
df['Post_Length']=df['posts'].str.len() + 1
sb.set_style('whitegrid')
sb.displot(df['Post_Length'],kde=False,bins=45)
plt.show()

#Dispaying the wordcloud
fig, ax = plt.subplots(len(df['type'].unique()), sharex=True, figsize=(15,len(df['type'].unique())))
k = 0
for i in df['type'].unique():
    df_4 = df[df['type'] == i]
    wordcloud = WordCloud(max_words=1628,relative_scaling=1,normalize_plurals=False,background_color="white").generate(df_4['posts'].to_string())
    plt.subplot(4,4,k+1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(i)
    ax[k].axis("off")
    k+=1
plt.show()



#PREPROCESSING

#Lowercasing the posts
df['posts'] = df['posts'].str.lower()
print(df)

#Remove URLs
df['posts'] = df['posts'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True).replace(r'https\S+', '', regex=True)
df['posts'] = df['posts'].str.replace('\W', ' ', regex=True)
print(df)

#Remove underscores
df['posts'] = df['posts'].str.replace('_', ' ', regex=True)
print(df)

#Remove leading and trailing whitespaces 
df['posts'] = df['posts'].str.strip()
print(df)

#Remove stop words in NLTK package
stop_words = stopwords.words('english')
df['posts'] = df['posts'].apply(lambda x: ' '.join([word
   for word in x.split() if word not in (stop_words)
]))
print(df)

#Format decimals to strings
df['posts'] = df['posts'].str.replace('\d+', '')
print(df)

#Tokenize the posts 
df['tokenized'] = [word_tokenize(entry) for entry in df['posts']]
print(df)

#Lemmatize the posts(Convert to the root form)
def lemmatize(s):
    s = [WordNetLemmatizer().lemmatize(word) for word in s]
    return s
df = df.assign(lemmatized = df.tokenized.apply(lambda x: lemmatize(x)))
df['lemmatizedstring'] = df['lemmatized'].apply(lambda x: ' '.join(map(str, x)))
print(df)

#Splitting the dataset into training and testing
train_data,test_data=train_test_split(df,test_size=0.2,random_state=42,stratify=df.type)

#Apply TF-IDF Vectorization
vectorizer=sklearn.feature_extraction.text.TfidfVectorizer(max_features=603,stop_words='english')
vectorizer.fit(train_data.lemmatizedstring)
train_post=vectorizer.transform(train_data.lemmatizedstring).toarray()
test_post=vectorizer.transform(test_data.lemmatizedstring).toarray()

#Label Encoder
target_encoder=LabelEncoder()
train_target=target_encoder.fit_transform(train_data.type)
test_target=target_encoder.fit_transform(test_data.type)

#Applying SMOTE to increase minority samples
sm = SMOTE(random_state=42)
train_res,train_target_res= sm.fit_resample(train_post,train_target)
print(train_res)


#Apply Logistic Regression and print the accuracy
from sklearn.linear_model import LogisticRegression
model_logreg=LogisticRegression(max_iter=3000,C=0.5,n_jobs=-1)
model_logreg.fit(train_res,train_target_res)
pred_lg=model_logreg.predict(test_post)
pred_training_lg=model_logreg.predict(train_res)
print("The train accuracy score for model trained on Logistic Regression is:",accuracy_score(train_target_res,pred_training_lg))
print("The test accuracy score for model trained on Logistic Regression is:",accuracy_score(test_target,pred_lg))

#Print the classification reports of Logistic Regression
print('train classification report \n ',sklearn.metrics.classification_report(train_target_res,model_logreg.predict(train_res),target_names=target_encoder.inverse_transform([i for i in range(16)])))
print('test classification report \n ',sklearn.metrics.classification_report(test_target,model_logreg.predict(test_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))

#Apply SVM and print the accuracy
model_svc=SVC()
model_svc.fit(train_res,train_target_res)
pred_svc=model_svc.predict(test_post)
pred_training_svc=model_svc.predict(train_res)
print("The train accuracy score for model trained on Support Classifier is:",accuracy_score(train_target_res,pred_training_svc))
print("The test accuracy score for model trained on Support Vector classifier is:",accuracy_score(test_target,pred_svc))

#Print the classification reports of SVC
print('train classification report \n ',sklearn.metrics.classification_report(train_target_res,model_svc.predict(train_res),target_names=target_encoder.inverse_transform([i for i in range(16)])))
print('test classification report \n ',sklearn.metrics.classification_report(test_target,model_svc.predict(test_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))

#Apply Naive Bayes and print the accuracy
model_multinomial_nb=MultinomialNB()
model_multinomial_nb.fit(train_res,train_target_res)
pred_nb=model_multinomial_nb.predict(test_post)
pred_training_nb=model_multinomial_nb.predict(train_res)
print("The train accuracy score for model trained on Naive Bayes is:",accuracy_score(train_target_res,pred_training_nb))
print("The test accuracy score for model trained on Naive Bayes is:",accuracy_score(test_target,pred_nb))

#Print the classification reports of Naive Bayes
print('train classification report \n ',sklearn.metrics.classification_report(train_target_res,model_multinomial_nb.predict(train_res),target_names=target_encoder.inverse_transform([i for i in range(16)])))
print('test classification report \n ',sklearn.metrics.classification_report(test_target,model_multinomial_nb.predict(test_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))

#Apply XGBoost and print the accuracy 
model_xgb=XGBClassifier(gpu_id=0,tree_method='gpu_hist',max_depth=5,n_estimators=50,learning_rate=0.2)
model_xgb.fit(train_res,train_target_res)
pred_xg=model_xgb.predict(test_post)
pred_training_xg=model_xgb.predict(train_res)
print("The train accuracy score for model trained on xgb is:",accuracy_score(train_target_res,pred_training_xg))
print("The test accuracy score for model trained on xgb is:",accuracy_score(test_target,pred_xg))

#Print the classification reports of XGBoost
print('train classification report \n ',sklearn.metrics.classification_report(train_target_res,model_xgb.predict(train_res),target_names=target_encoder.inverse_transform([i for i in range(16)])))
print('test classification report \n ',sklearn.metrics.classification_report(test_target,model_xgb.predict(test_post),target_names=target_encoder.inverse_transform([i for i in range(16)])))

#Applying 1d-CNN and get the results
X_train = np.array(train_res)
X_test = np.array(test_post)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], 1)
batch_size = 32
num_classes = 16
epochs = 50
input_shape = (X_train.shape[1], 1)

#Design the CNN model
model = Sequential()
intput_shape=(X_train.shape[1], 1)
model.add(Conv1D(64, kernel_size=3,padding = 'same',activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=(2)))
model.add(Conv1D(64,kernel_size=3,padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=(2)))
model.add(Conv1D(32,kernel_size=3,padding = 'same', activation='relu'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

#Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

Y_train = np.array(train_target_res)
Y_test = np.array(test_target)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
Y_test.shape

history=model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,  
          verbose=1,validation_data=(X_test,Y_test))

#Evaluate the model
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_pred=model.predict(X_test)
Y_test1=np.argmax(Y_test, axis=1)
y_pred=np.argmax(y_pred,axis=1)

#Plot the confusion matrix
cmatrix=confusion_matrix(Y_test1, y_pred)
figure = plt.figure(figsize=(8, 8))
sb.heatmap(cmatrix, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#Find the cross validation accuracy for SVM
model_svc=SVC()
kf = KFold(n_splits=10, shuffle=True, random_state=1)

acc_train_score = []
acc_test_score = []

for train_index, test_index in kf.split(df):
    train_data, test_data = df.iloc[train_index, :], df.iloc[test_index, :]

    vectorizer.fit(train_data.lemmatizedstring)
    train_post=vectorizer.transform(train_data.lemmatizedstring).toarray()
    test_post=vectorizer.transform(test_data.lemmatizedstring).toarray()

    train_target=target_encoder.fit_transform(train_data.type)
    test_target=target_encoder.fit_transform(test_data.type)

    train_res,train_target_res= sm.fit_resample(train_post,train_target)

    model_svc.fit(train_res,train_target_res)
    pred_svc=model_svc.predict(test_post) 
    pred_training_svc=model_svc.predict(train_res)

    acc_train = accuracy_score(train_target_res, pred_training_svc)
    acc_test = accuracy_score(test_target, pred_svc)
    acc_train_score.append(acc_train)
    acc_test_score.append(acc_test)
    print(acc_test_score)

avg_acc_train_score = sum(acc_train_score)/10
avg_acc_test_score = sum(acc_test_score)/10

print("Average train accuracy: {}".format(avg_acc_train_score))
print("Average test accuracy: {}".format(avg_acc_test_score)) 

#Find the cross validation accuracy for XGB
model_xgb = XGBClassifier(gpu_id=0,tree_method='gpu_hist',max_depth=5,n_estimators=50,learning_rate=0.2)
kf = KFold(n_splits=10, shuffle=True, random_state=1)

acc_train_score = []
acc_test_score = []

for train_index, test_index in kf.split(df):
    train_data, test_data = df.iloc[train_index, :], df.iloc[test_index, :]

    vectorizer.fit(train_data.lemmatizedstring)
    train_post=vectorizer.transform(train_data.lemmatizedstring).toarray()
    test_post=vectorizer.transform(test_data.lemmatizedstring).toarray()

    train_target=target_encoder.fit_transform(train_data.type)
    test_target=target_encoder.fit_transform(test_data.type)

    train_res,train_target_res= sm.fit_resample(train_post,train_target)

    model_xgb.fit(train_res,train_target_res)
    pred_xg=model_xgb.predict(test_post) 
    pred_training_xg=model_xgb.predict(train_res)

    acc_train = accuracy_score(train_target_res, pred_training_xg)
    acc_test = accuracy_score(test_target, pred_xg)
    acc_train_score.append(acc_train)
    acc_test_score.append(acc_test)
    print(acc_test_score)

avg_acc_train_score = sum(acc_train_score)/10
avg_acc_test_score = sum(acc_test_score)/10

print("Average train accuracy: {}".format(avg_acc_train_score))
print("Average test accuracy: {}".format(avg_acc_test_score))


#Find the cross validation accuracy for Naive Bayes
model_multinomial_nb=MultinomialNB()
kf = KFold(n_splits=10, shuffle=True, random_state=1)

acc_train_score = []
acc_test_score = []

for train_index, test_index in kf.split(df):
    train_data, test_data = df.iloc[train_index, :], df.iloc[test_index, :]

    vectorizer.fit(train_data.lemmatizedstring)
    train_post=vectorizer.transform(train_data.lemmatizedstring).toarray()
    test_post=vectorizer.transform(test_data.lemmatizedstring).toarray()

    train_target=target_encoder.fit_transform(train_data.type)
    test_target=target_encoder.fit_transform(test_data.type)

    train_res,train_target_res= sm.fit_resample(train_post,train_target)

    model_multinomial_nb.fit(train_res,train_target_res)
    pred_nb=model_multinomial_nb.predict(test_post)
    pred_training_nb=model_multinomial_nb.predict(train_res)

    acc_train = accuracy_score(train_target_res,pred_training_nb)
    acc_test = accuracy_score(test_target,pred_nb)
    acc_train_score.append(acc_train)
    acc_test_score.append(acc_test)
    print(acc_test_score)

avg_acc_train_score = sum(acc_train_score)/10
avg_acc_test_score = sum(acc_test_score)/10

print("Average train accuracy: {}".format(avg_acc_train_score))
print("Average test accuracy: {}".format(avg_acc_test_score))


#Find the cross validation accuracy for Logistic Regression
model_logreg=LogisticRegression(max_iter=3000,C=0.5,n_jobs=-1)
kf = KFold(n_splits=10, shuffle=True, random_state=1)

acc_train_score = []
acc_test_score = []


for train_index, test_index in kf.split(df):
    train_data, test_data = df.iloc[train_index, :], df.iloc[test_index, :]

    vectorizer.fit(train_data.lemmatizedstring)
    train_post=vectorizer.transform(train_data.lemmatizedstring).toarray()
    test_post=vectorizer.transform(test_data.lemmatizedstring).toarray()

    train_target=target_encoder.fit_transform(train_data.type)
    test_target=target_encoder.fit_transform(test_data.type)

    train_res,train_target_res= sm.fit_resample(train_post,train_target)

    model_logreg.fit(train_res,train_target_res)
    pred_lg=model_logreg.predict(test_post)
    pred_training_lg=model_logreg.predict(train_res)

    acc_train = accuracy_score(train_target_res,pred_training_lg)
    acc_test = accuracy_score(test_target,pred_lg)
    acc_train_score.append(acc_train)
    acc_test_score.append(acc_test)
    print(acc_test_score)

avg_acc_train_score = sum(acc_train_score)/10
avg_acc_test_score = sum(acc_test_score)/10

print("Average train accuracy: {}".format(avg_acc_train_score))
print("Average test accuracy: {}".format(avg_acc_test_score))