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

    #step 3 : Scaled the data
    scaler = StandardScaler()
    X_Scaled = scaler.fit_transform(X)

    print("Scaled Data : ")
    print(X_Scaled[:5])    

    #Step 4 : Elbow Method
    WCSS = list()

    for k in range(1 , 11):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_Scaled)
        WCSS.append(model.inertia_)

    print("Values of WCSS : ")
    for i in range(len(WCSS)):
        print(f"{i+1} : {WCSS[i]}")

    #Step 5 : Visualition
    plt.plot(range(1,11) , WCSS , marker="o")
    plt.xlabel("No. of clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")

    plt.grid()
    plt.show()

    #Step 6 : Final model
    model = KMeans(
                n_clusters=4,
                random_state=42,
                n_init=10
            )
    clusters = model.fit_predict(X_Scaled)
    df["Cluster"] = clusters

    print("Dataset with clusters : ")
    print(df.tail(100))

if __name__ == "__main__":
    main()