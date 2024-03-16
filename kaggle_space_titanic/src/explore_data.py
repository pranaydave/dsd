import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px

file_train = '/Users/pd186029/Documents/Pranay/Teradata/Development/full_stck/data_jarvis_data/data_insights_for_researchers/space_titanic/train.csv'
df = pd.read_csv(file_train,sep=',')
df.head(3)
print (df.head(3))

##Statistics
stats = df.describe()
numeric_columns = list(stats.columns)
for col in numeric_columns:
    fig = px.line(stats, x=stats.index, y=stats[col], title=col,height=300)
    fig.show()

