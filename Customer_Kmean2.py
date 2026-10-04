import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    #step 1 : load data
    df = pd.read_csv("Mall_Customers.csv")
    print("Data set loaded")
    print(df.head())

    print("Missing values ")
    print(df.isnull().sum())

    #step 2 : Feature seclection
    X = df[["AnnualIncome" , "SpendingScore"]]
    print("Selected features : ")
    print(X.head())
    
    

if __name__ == "__main__":
    main()