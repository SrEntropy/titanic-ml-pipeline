
import pandas as pd

def clean_titanic_data(df):
    df.drop(columns=["PassengerId", "Cabin"], inplace = True)
    df["Age"] = df[["Age"]].fillna(df["Age"].mean())
    df["Sex"] = df["Sex"].map({"":["male":0, female:1}]
    df = pd.get_dummies(columns = ["Embarked"], drop_first = True)
    return df
                            


file  = pd.read_csv("Path to File")
clean_data = clean_titanic_data(file)


