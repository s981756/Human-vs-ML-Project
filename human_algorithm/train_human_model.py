import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from human_algorithm.human_classifier import human_classify
from data.get_data import load_student_data
from sklearn.model_selection import train_test_split


# This section of code separates the whole data-set into training and testing data.
df, df['passing?'] = load_student_data()
train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df['passing?']
)

# This section of code applies the human classification algorithm to the test data.
test_df['human_prediction'] = test_df['studytime'].apply(human_classify)
test_df['correct'] = test_df['human_prediction'] == test_df['passing?']
accuracy = (test_df['human_prediction'] == test_df['passing?']).mean()
print(f"Human classifier accuracy: {accuracy:.2%}")

# Here we print the confusion matrix to see how well the human classifier performed on the test-data subset.
conf_matrix = pd.crosstab(
    test_df['passing?'],
    test_df['human_prediction'],
    rownames=['Actual'],
    colnames=['Predicted']
)
print(conf_matrix)

# Finally, we print one example of a failure case where the human classifier got the prediction wrong.
failure_row = test_df[test_df['human_prediction'] != test_df['passing?']].iloc[0]
print("\nFAILURE EXAMPLE")
print(failure_row[['studytime', 'famrel', 'passing?', 'human_prediction']])


# Print a scatter plot showing correct vs incorrect predictions.
os.makedirs("human_algorithm/plots", exist_ok=True)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=test_df,
    x='studytime',
    y='famrel',
    hue='correct',
    style='correct',
    s=100,
    palette={True: 'green', False: 'red'}
)

plt.title('Human Algorithm: Correct vs Incorrect Predictions')
plt.xlabel('study time (hours per week)')
plt.ylabel('family relationship score (1-5)')
plt.legend(title='Prediction Correct')
plt.grid(True)
plt.savefig('human_algorithm/plots/human_model_training_results.png', dpi=150)
plt.close()