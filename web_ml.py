import pandas as pd
import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Giving a Title
st.title("Streamlit Web App")

# writing text in app
st.write("""
# Explore different classifier
find which one is best
""")

# Adding selectbox
# dataset_name = st.selectbox("Select Dataset",("Iris","Breast Cancer","Wine Dataset"))
# st.write(dataset_name)

# creating side bar for dataset
dataset_name = st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Wine Dataset"))

# creating sidebar for classifier
classifier_name = st.sidebar.selectbox("select classifier",("KNN","SVM","Random Forest"))

# getting dataset and return x&y
def get_dataset(dataset_name):
    if dataset_name=="Iris":
        data= datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        data= datasets.load_breast_cancer()
    else:
        data= datasets.load_wine()

    x = data.data
    y = data.target
    return x, y

x, y = get_dataset(dataset_name)
st.write("shape of dataset", x.shape)
st.write("number of classes", len(np.unique(y)))


def add_parameter_ui(clf_name):
    params=dict()
    if clf_name == "KNN":
        k= st.sidebar.slider("k", 1, 15)
        params["k"]= k
    elif clf_name == "SVM":
        c = st.sidebar.slider("c", .01, 10.0)
        params["c"]=c
    else:
        max_depth = st.sidebar.slider("max_depth",2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"]= max_depth
        params["n_estimators"]= n_estimators
    return  params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["k"])
    elif clf_name == "SVM":
        clf = SVC(c=params["c"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"])

    return clf

clf = get_classifier(classifier_name,params)


# classification
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.2, random_state=1)

clf.fit(x_train,y_train)
y_pred= clf.predict(x_test)

acc = accuracy_score(y_test,y_pred)
st.write(f"classifier={classifier_name}")
st.write(f"accuracy={acc}")

# plotting the dataset
pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:,0]
x2 = x_projected[:,1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.colorbar()
st.pyplot()
#plt.show()

