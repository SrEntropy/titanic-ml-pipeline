
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data_processing import DataPreprocessing

df = pd.read_csv("data/Titanic-Dataset.csv")
processData = DataPreprocessing(df)
processData.clean_data()
processData.feature_engineering()
processData.pre_normaliztion("Fare")
processData.feature_normalization(['Age', 'Fare', 'SibSp', 'Parch','FamilySize'])
processData.encode_gender()
processData.one_hot_encoding_features(["Embarked", "AgeGroup", "TitleGroup"])
processData.drop_features(["PassengerId", "Name", "Title", "Ticket", "Cabin"])
processData.data.head(3)
#processData.data.isna().sum()

processData