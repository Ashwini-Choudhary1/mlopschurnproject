import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

raw_data_path = Path("data/raw/customerchurn.csv")
processed_path = Path("data/processed")

numeric_cols =["tenure","MonthlyCharges","TotalCharges"]
cat_cols=["Contract"]
target_col = "Churn"

def preprocess():

    df = pd.read_csv(raw_data_path) # read csv

    df = df[numeric_cols + cat_cols + [target_col]] # col needed for df

    #convert numeric columns

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors='coerce')
    df['tenure']= df['tenure'].astype(int)
    df['MonthlyCharges'] = df['MonthlyCharges'].astype('float')

    df= df.dropna()

    #encode the categorical columns

    encoders={}
    for col in cat_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

    # encode the target
    target_encoder=LabelEncoder()
    df[target_col]= target_encoder.fit_transform(df[target_col])

    X= df.drop(target_col,axis=1)
    y= df[target_col]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    processed_path.mkdir(parents=True,exist_ok=True)

    X_train.to_csv(processed_path/"Xtrain.csv",index= False)
    X_test.to_csv(processed_path / "X_test.csv", index=False)
    y_train.to_csv(processed_path / "y_train.csv", index=False)
    y_test.to_csv(processed_path / "y_test.csv", index=False)

    print("Data preprocessing completed successfully")

if __name__=="__main__":
    preprocess()


    