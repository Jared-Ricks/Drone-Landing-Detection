import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('obb_data.csv')

X = df.drop(columns=['Target', 'Name'])
Y = df['Target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=3)

svm = SVC(kernel='linear')
svm.fit(X_train, Y_train)
pred = svm.predict(X_test)

conf_mat = confusion_matrix(Y_test, pred)
conf_mat = np.array(conf_mat)
accuracy = accuracy_score(Y_test, pred) * 100
print('Classification Report:\n',classification_report(Y_test, pred))
print(f'Accuracy: {accuracy:.2f}%')
print('\nConfusion Matrix:\n', conf_mat)
fig, ax = plt.subplots(figsize=(7, 5)) 
sns.heatmap(conf_mat/np.sum(conf_mat), annot=True, fmt= '.2%', cmap='Blues', 
            xticklabels=['Misaligned' , 'Aligned'], yticklabels=['Misaligned', 'Aligned'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
