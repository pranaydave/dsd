from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score,roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

test_size = 0.3
        
##random_state=42 (any constant value) will always give the same test train split
dfX_train, dfX_test, dfy_train, dfy_test = train_test_split(df_train_mvrepl_rmoutlier, df_train1[target_column], test_size=test_size, random_state=0)

pipeline = Pipeline([
('clf', GradientBoostingClassifier())
])


# Training the model
pipeline.fit(dfX_train, dfy_train)
# Making predictions
y_pred = pipeline.predict(dfX_test)
# Evaluating the model
accuracy = accuracy_score(dfy_test, y_pred)
print("Accuracy:", accuracy)

# Create a confusion matrix
conf_matrix = confusion_matrix(dfy_test, y_pred)

# Plot the confusion matrix using seaborn and matplotlib
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()

# Make predictions on the test set
y_pred_prob = pipeline.predict_proba(dfX_test)[:, 1]  # Predicted probabilities for class 1

# Calculate AUC-ROC score
auc_score = roc_auc_score(dfy_test, y_pred_prob)
print ('auc_score..',auc_score)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(dfy_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


