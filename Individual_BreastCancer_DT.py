import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

def main():
    #step 1 : Load the dataset

    df =  pd.read_csv("breast_cancer.csv")
    print("shape of Dataset : ",df.shape)
    print("First five records : ")
    print(df.head())

    #Step 2 : Seperate features and ladels

    X = df.drop("target" , axis=1)
    Y = df["target"]

    print("X shape : ",X.shape)
    print("Y shape : ",Y.shape)

    #Step 3 : Split dataset for training and testing

    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42)

    #step 4 : Scale the features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)


    #Step 5 : Create the model
    model = DecisionTreeClassifier(random_state=42)

    #Step 6 : train the model
    model.fit(X_train , Y_train)

    #Step 7 : Test the model
    Y_pred = model.predict(X_test)

    #Step 8 : Evalaute the model

    print("Accuracy : ",accuracy_score(Y_test , Y_pred))

    print("Confisuion matrix : ")
    print(confusion_matrix(Y_test , Y_pred))



if __name__ == "__main__":
    main()