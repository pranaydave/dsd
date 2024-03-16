import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import matplotlib.pyplot as plt
%matplotlib inline


file_dir = ''
file_train = file_dir + 'train.csv'
df = pd.read_csv(file_train,sep=',')
df.head(3)
print (df.head(3))

##Descriptive Statistics - Numeric cols
stats = df.describe()
numeric_columns = list(stats.columns)
for col in numeric_columns:
    fig = px.line(stats, x=stats.index, y=stats[col], title=col,height=300)
    fig.show()
    
##descriptive Statistics - Categorical varaibles

non_numeric_cols = ['PassengerId',
 'HomePlanet',
 'CryoSleep',
 'Cabin',
 'Destination',
 'VIP',
 'Name',
 'Transported']

for col in non_numeric_cols:
    unique_value = df[col].value_counts()
    col_keys = unique_value.keys()
    col_val = unique_value.values
    fig = px.bar( x=col_keys, y=col_val,title=col,
                height=300)
    fig.show()

##Find NAN
nan_counts = df.isna().sum()
print(nan_counts)

