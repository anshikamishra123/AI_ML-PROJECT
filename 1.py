import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, confusion_matrix,
                             precision_score, f1_score, classification_report, roc_curve)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load dataset
df = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Basic data inspection
df.head()
df.shape
df.info()
df.columns.values
df.dtypes

# Visualize missing values
msno.matrix(df)

# Data cleaning
df = df.drop(['customerID'], axis=1)
df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)
df.fillna({"TotalCharges": df["TotalCharges"].mean()}, inplace=True)
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_cols]
