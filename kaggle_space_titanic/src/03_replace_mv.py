from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder
import pandas as pd
from sklearn.impute import KNNImputer



df_train_trf.isna().any()
df_test_trf.isna().any()

#columns_mvrepl = ['Age','HomePlanet',  'CryoSleep', 'Destination', 'VIP','Cabin0','Cabin1', 'Cabin2']
#columns_mvrepl_no = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','PassengerId0','PassengerId1']
columns_mvrepl_names= [x+'_mvrepl' for x in df_train_trf.columns]


# Apply KNNImputer
imputer = KNNImputer(n_neighbors=5)  # Choose the number of neighbors
df_train_mvrepl = pd.DataFrame(imputer.fit_transform(df_train_trf),columns=df_train_trf.columns)
df_test_mvrepl = pd.DataFrame(imputer.transform(df_test_trf),columns=df_test_trf.columns)

df_train_mvrepl.isna().any()
df_test_mvrepl.isna().any()

list(df_train_mvrepl.columns)