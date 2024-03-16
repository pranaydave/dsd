from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import plotly.express as px

# Initialize PCA object
pca = PCA(n_components=2)  # Specify the number of components (dimensionality of the reduced data)

# Fit the data and transform it to the reduced dimensionality
reduced_data = pca.fit_transform(df_train_mvrepl)
df_train_pca = pd.DataFrame(reduced_data)
df_train_pca.columns = ['dim0','dim1']



#Isolation forest
contamination = 0.005
clf = IsolationForest(random_state=0,contamination=contamination)
df_train_pca['outlier'] = clf.fit_predict(df_train_pca)

# Create a scatter plot
fig = px.scatter(df_train_pca, x='dim0', y='dim1', color='outlier', title='PCA', labels={'x': 'dim0', 'y': 'dim1'})

# Show the plot
fig.show()

df_train_pca[df_train_pca['outlier']==-1].shape

##Adjust df_train_mvrepl
df_train_mvrepl_rmoutlier = df_train_mvrepl.copy()
df_train1 = df_train.copy()

df_train_mvrepl_rmoutlier['outlier'] = df_train_pca['outlier'] 
df_train1['outlier'] = df_train_pca['outlier'] 

df_train_mvrepl_rmoutlier = df_train_mvrepl_rmoutlier[df_train_mvrepl_rmoutlier['outlier']==1]
df_train1 = df_train1[df_train1['outlier']==1]

df_train_mvrepl_rmoutlier = df_train_mvrepl_rmoutlier.drop(columns=['outlier'])
df_train1 = df_train1.drop(columns=['outlier'])

df_train_mvrepl.shape
df_train_mvrepl_rmoutlier.shape
df_train1.shape