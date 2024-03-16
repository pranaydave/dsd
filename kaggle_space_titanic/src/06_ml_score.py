

##Scoring
y_score = pipeline.predict(df_test_mvrepl)

# Convert the array to a DataFrame
df_test_out = df_test.copy()[['PassengerId']]
df_test_out['pred'] = y_score

# Save the DataFrame to a CSV file
file_out = file_dir + 'my_submission3.csv'
df_test_out.to_csv(file_out, index=False)
