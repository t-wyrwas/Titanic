import pandas as pd
import os

def get_train_test_matrices(df, y_id):
    from sklearn.model_selection import train_test_split
    X = df.drop(y_id, axis=1).values.astype('float')
    y = df.loc[:, y_id].ravel()
    return train_test_split(X, y, test_size=0.2, random_state=0)

def show_metrics(real, predictions):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
    print(f"Accuracy score: {accuracy_score(real, predictions):.2}")
    print(f"Confusion matrix: {confusion_matrix(real, predictions)}")
    print(f"Precision: {precision_score(real, predictions):.2}")
    print(f"Recall: {recall_score(real, predictions):.2}")

def create_submission_file(model, df_test, filename):
    predictions = model.predict(df_test)
    df_submission = pd.DataFrame({'PassengerId': df_test.index, 'Survived': predictions})
    submission_data_file = os.path.join('data', 'external', filename)
    df_submission.to_csv(submission_data_file, index=False)
