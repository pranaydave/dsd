import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder

file_train = file_dir +'train.csv'
file_test = file_dir +'test.csv'

df_train = pd.read_csv(file_train,sep=',')
df_test = pd.read_csv(file_test,sep=',')

df_train[['PassengerId0', 'PassengerId1']] = df_train['PassengerId'].str.split('_', expand=True)
df_test[['PassengerId0', 'PassengerId1']] = df_test['PassengerId'].str.split('_', expand=True)

df_train[['Cabin0', 'Cabin1','Cabin2']] = df_train['Cabin'].str.split('/', expand=True)
df_test[['Cabin0', 'Cabin1','Cabin2']] = df_test['Cabin'].str.split('/', expand=True)


df_train[['firstname', 'lastname']] = df_train['Name'].str.split(' ', expand=True)
df_test[['firstname', 'lastname']] = df_test['Name'].str.split(' ', expand=True)


categorical_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP','Cabin0','Cabin2']
numeric_columns = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','PassengerId0', 'PassengerId1','Cabin1']
target_column  = ['Transported']


# One hot encoding and srandard  scaler
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),  # StandardScaler for numeric feature
        ('cat', OrdinalEncoder(),categorical_columns)  # OneHotEncoder for categorical feature
    ])

preprocessor.fit(df_train)


#Recreate transformed dataframe
# Get the names of the transformed columns
numeric_feature_names = [col for col in preprocessor.transformers_[0][2]]
categorical_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns))
col_names = numeric_feature_names + categorical_feature_names


arr_train1 = preprocessor.transform(df_train)
arr_test1 = preprocessor.transform(df_test)

df_train_trf = pd.DataFrame(arr_train1,columns=col_names)
df_test_trf = pd.DataFrame(arr_test1,columns=col_names)





