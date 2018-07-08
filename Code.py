import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn import metrics
import seaborn as sns
from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Combine boths dataframe
train_df['Data'] = 'Train'
test_df['Data'] = 'Test'
both_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
both_df['subject'] = '#' + both_df['subject'].astype(str)

label = both_df.pop('Activity')
print(label)
print('Shape Train:\t{}'.format(train_df.shape))
print('Shape Test:\t{}\n'.format(test_df.shape))

train_df.head()
#Features which your Mobile knows abt u
print(pd.DataFrame.from_dict(Counter([col.split('-')[0].split('(')[0] for col in both_df.columns]), orient='index').rename(columns={0:'count'}).sort_values('count', ascending=False))
#type of  data we have
print(both_df.info())
label_counts = label.value_counts()


plt.scatter(label_counts.index,label_counts)
plt.show()


'''fig, axarr = plt.subplots(5, 6, figsize=(15,6))
for person in range(0, 30):
    single_person = both_df[(label == 'WALKING') & (both_df['subject'] == '#{}'.format(person + 1))].drop(
        ['subject', 'Data'], axis=1)
    scl = StandardScaler()
    tsne_data = scl.fit_transform(single_person)
    pca = PCA(n_components=0.9, random_state=3)
    tsne_data = pca.fit_transform(tsne_data)
    tsne = TSNE(random_state=3)
    tsne_transformed = tsne.fit_transform(tsne_data)

    axarr[person // 6][person % 6].plot(tsne_transformed[:, 0], tsne_transformed[:, 1], '.-')
    axarr[person // 6][person % 6].set_title('Participant #{}'.format(person + 1))
    axarr[person // 6][person % 6].axis('off')

plt.tight_layout()
plt.show()'''

'''array=train_df.values
X=train_df['Activity']
Y=train_df.drop('Activity',axis=1)

test_size=0.33
seed=6
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)

model=KNeighborsClassifier()
model.fit(X,Y)
print(model.score(X,Y))'''