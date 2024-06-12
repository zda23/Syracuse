#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Load the CSV file
test = '/Users/zanealderfer/Downloads/fashion-mnist_test.csv'
test = pd.read_csv(test)
train = '/Users/zanealderfer/Downloads/fashion-mnist_train.csv'
train = pd.read_csv(train)

# Display the first few rows of the dataframe
test.head()


# In[3]:


train.head()


# In[22]:


import matplotlib.pyplot as plt


# Developed an initial analysis of the labels in the test file.  Found the labels are very evenly distributed and there are no NULLs within the data. 

# In[23]:


# Step 2: Explore the features
print("Dataset shape:", test.shape)
print("Columns:", test.columns)

# Step 3: Identify the target variable
target_variable = 'label'
print("Target variable:", target_variable)

# Step 4: Check for missing values
print("Missing values:", test.isnull().sum())

# Step 5: Explore the distribution of the target variable
plt.figure(figsize=(8, 6))
test[target_variable].value_counts().sort_index().plot(kind='bar')
plt.title("Distribution of Label")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.show()


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Developing a Naive Bayes Classifier on the testing dataset to identify the accuracy of a Naive Bayes classifier

# In[5]:


X = test.drop('label', axis=1)  # Features (pixel values)
y = test['label']  # Target labels (digits)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Step 5: Evaluate the classifier
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# You can also print classification report for more detailed evaluation
print(classification_report(y_test, y_pred))


# In[6]:


X = train.drop('label', axis=1)  # Features (pixel values)
y = train['label']  # Target labels (digits)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Step 5: Evaluate the classifier
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# You can also print classification report for more detailed evaluation
print(classification_report(y_test, y_pred))


# In[14]:


from sklearn.ensemble import GradientBoostingClassifier


# Running a Boosting Tree Classifier model on both the test and training data again to see which dataset produces the best classifier percent

# In[15]:


# Step 2: Prepare the data
X = test.drop('label', axis=1)  # Features (pixel values)
y = test['label']  # Target labels (digits)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and train the Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb_classifier.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = gb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# You can also print classification report for more detailed evaluation
print(classification_report(y_test, y_pred))


# In[18]:


from sklearn.linear_model import LogisticRegression


# I will develop a linear classification model as seen below.  The results of this model will be compared to the other two classifiers for accuracy.

# In[20]:


# Step 2: Prepare the data
X = test.drop('label', axis=1)  # Features (pixel values)
y = test['label']  # Target labels (digits)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and train the Logistic Regression model
logistic_classifier = LogisticRegression(max_iter=5000)  # Adjust max_iter as needed
logistic_classifier.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = logistic_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# You can also print classification report for more detailed evaluation
print(classification_report(y_test, y_pred))


# I used 3 types of classifier models: Naive Bayes classifier, Boosting Tree classifier, and a Linear Classification model.  The Naive Bayes model is simple and easy to implement and doesn't require much training data while also being highly scalable.  Boosting tree models tend to be more accurate but also train faster on larger datasets.  A linear classifier is very simple and has a lot of computational attractiveness and easy to implement. 
# With that being said, the Boosting Tree classifier delivered the most promising results with a 0.851 accuracy result for the labels.  This was followed by the linear classifier with 0.753 and then the Naive Bayes classifier with a 0.684.  I only ran the Naive Bayes classifier on the training and testing data however.  I find a slightly larger accuracy in the test data for Naive Bayes than the train data.  I only ran the Boosting Tree and Linear classifiers on the test data as they were taking a long time to generate results on the training model for some reason.
