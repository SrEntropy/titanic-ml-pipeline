import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> pd.DataFrame:
  #Load raw titanic dataset from CSV.
  df = pd.read_csv(file_path)
  assert df.shape[0] > 0, "Dataset is empty"
  assert "Survived" in df.columns, "Missing target column 'Survived'"
  return df
  
def split_data(df: pd.DataFrame, target: str = "Survived", test_size: float = 0.2, random_state: int = 42):
  """Split dataset into training and test sets
  #Before preprocessing"""
  X = df.drop(columns=[target])
  y = df[target]
  return train_test_split(X, y, test_size=test_size, random_state=random_state)
