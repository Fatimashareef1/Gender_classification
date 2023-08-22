#!/usr/bin/env python
# coding: utf-8

# ### Gender Classification Model Using Machine Learning 

# In[65]:


import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import time


# In[3]:


train_female = '/Users/HP/gender/train/female'
train_male   = '/Users/HP/gender/train/male'
test_female  = '/Users/HP/gender/test/female'
test_male    = '/Users/HP/gender/test/male'


# In[4]:


Image.open('/Users/HP/gender/test/male/001.jpg')


# In[5]:


image_size = 64
for image1 in tqdm(os.listdir(train_female)):
    path = os.path.join(train_female, image1)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (image_size, image_size)).flatten()
    np_img = np.asarray(img)
    
for image2 in tqdm(os.listdir(train_male)):
    path2 = os.path.join(train_male, image2)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.resize(img2, (image_size, image_size)).flatten()
    np_img2 = np.asarray(img2)


# In[6]:


image_size = 64
for image1 in tqdm(os.listdir(test_female)):
    path = os.path.join(test_female, image1)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (image_size, image_size)).flatten()
    np_img = np.asarray(img)
    
for image2 in tqdm(os.listdir(test_male)):
    path2 = os.path.join(test_male, image2)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.resize(img2, (image_size, image_size)).flatten()
    np_img2 = np.asarray(img2)


# In[7]:


train_data_female = []
train_data_male = []
for image1 in tqdm(os.listdir(train_female)):
    path = os.path.join(train_female, image1)
    img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (image_size, image_size))
    train_data_female.append(img1)
        
for image2 in tqdm(os.listdir(train_male)): 
    path2 = os.path.join(train_male, image2)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE) 
    img2 = cv2.resize(img2, (image_size, image_size))
    train_data_male.append(img2) 
    
Train_data = np.concatenate((np.asarray(train_data_female), np.asarray(train_data_male)), axis = 0)


# In[46]:


x_train.shape


# In[8]:


test_data_female = []
test_data_male = []
    
for image1 in tqdm(os.listdir(test_female)): 
    path = os.path.join(test_female, image1)
    img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    img1 = cv2.resize(img1, (image_size, image_size))
    test_data_female.append(img1)
        
for image2 in tqdm(os.listdir(test_male)): 
    path2 = os.path.join(test_male, image2)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE) 
    img2 = cv2.resize(img2, (image_size, image_size))
    test_data_male.append(img2) 
    
test_data= np.concatenate((np.asarray(test_data_female),np.asarray(test_data_male)),axis=0) 
   


# In[9]:


X_data = np.concatenate((Train_data, test_data), axis = 0)
X_data


# In[10]:


X_data = (X_data - np.min(X_data))/(np.max(X_data) - np.min(X_data))
X_data.shape


# ### Labels

# In[11]:


zeros = np.zeros(1747)
ones  = np.ones(1744)
y_train=np.concatenate((zeros,ones),axis = 0)
y_train
zero1 = np.zeros(100)
one1  = np.ones(100)
y_test =np.concatenate((zero1,one1),axis =0)


# In[12]:


y_data =np.concatenate((y_train,y_test),axis = 0).reshape(X_data.shape[0],1)


# In[13]:


y_data.shape
X_data.shape


# In[14]:


x_train,x_test,y_train,y_test = train_test_split(X_data , y_data , random_state = 42 , test_size= 0.2 )


# In[15]:


number_of_train = x_train.shape[0]
number_of_test = x_test.shape[0]


# In[16]:


x_train_flatten = x_train.reshape(number_of_train, x_train.shape[1] * x_train.shape[2])
x_test_flatten = x_test.reshape(number_of_test, x_test.shape[1] * x_test.shape[2])
print("X train flatten",x_train_flatten.shape)
print("X test flatten",x_test_flatten.shape)


# In[17]:


x_train = x_train_flatten
x_test = x_test_flatten
y_test = y_test.T.flatten()
y_train = y_train.T.flatten()


# In[18]:


print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)


# In[81]:


# Create a list of classifier instances
classifiers = [
    ("Complement Naive Bayes", ComplementNB()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier()),
    ("LogisticRegression", LogisticRegression()),
    ("SVC", SVC()),
    ("KNeighborsClassifier", KNeighborsClassifier())
    
]

# Initialize lists to store classifier names, training accuracies, and test accuracies
classifier_names = []
train_accuracies = []
test_accuracies = []
training_times = []

# Iterate through the classifiers and evaluate their performance
for name, classifier in classifiers:
    start_time = time.time()  # Start timing
    classifier.fit(x_train, y_train)
    training_time = time.time() - start_time
    y_pred = classifier.predict(x_test)
    train_score = classifier.score(x_train, y_train)
    test_score = classifier.score(x_test, y_test)
    
    print(f"Classifier: {name}")
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Training Score: {train_score*100:.4f}")
    print(f"Testing Score: {test_score*100:.4f}")
    
    # Append classifier name, training accuracy, and test accuracy to lists
    classifier_names.append(name)
    
    train_accuracies.append(train_score)
    test_accuracies.append(test_score)
    training_times.append(training_time)
    # Calculate confusion matrix
    conmat = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix as heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(conmat, annot=True, fmt=".1f", linewidth=1, cmap="crest")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix for {name}")
    plt.show()
    
    # Calculate and print classification report
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(class_report)
    
    print("-" * 120)



# In[82]:


# Create a bar plot of training and test accuracies
plt.figure(figsize=(12, 8))
bar_width = 0.35  # Width of the bars
indices = range(len(classifier_names))

plt.bar(indices, train_accuracies, bar_width, label='Training Accuracy', color='skyblue')
plt.bar([i + bar_width for i in indices], test_accuracies, bar_width, label='Test Accuracy', color='orange')

plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracies of Classifiers")
plt.xticks([i + bar_width / 2 for i in indices], classifier_names, rotation=45)
plt.ylim(0, 1.0)  # Set y-axis limits
plt.legend()
plt.tight_layout()
plt.show()


# In[83]:


# Create a bar plot of training times for each classifier
plt.figure(figsize=(12, 8))
plt.bar(classifier_names, training_times, color='green')
plt.xlabel("Classifier")
plt.ylabel("Training Time (seconds)")
plt.title("Training Times of Classifiers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### Deployement

# In[38]:


path = '002.jpg'
user_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
user_img = cv2.resize(user_img, (64, 64)).flatten()
np_user_img = np.asarray(img)


# In[60]:


dep = (np_user_img - np.min(np_user_img))/(np.max(np_user_img) - np.min(np_user_img))
dep1=dep.reshape(np_user_img.shape[0],1).T
dep1.shape


# In[53]:


dep.shape
x_train.shape


# ###  ' 0 ' for female and ' 1 ' for male 

# In[75]:


y_pred1=model.predict(dep1)
y_pred1


# In[78]:


path11 = '002.jpg'
image11 = cv2.imread(path11, cv2.IMREAD_GRAYSCALE)
resize11 = cv2.resize(image11, (64, 64)).flatten().reshape(1, -1)
prediction11 = model.predict(resize11)
prediction11
cv2.imshow('Resized Image with Prediction: ' + str(prediction11), image11)
cv2.waitKey(0)
cv2.destroyAllWindows()

