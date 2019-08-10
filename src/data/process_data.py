import numpy as np
import pandas as pd
import os

def read_data():
    raw_data_path = os.path.join(os.path.pardir, 'data', 'raw')
    train_file_path = os.path.join(raw_data_path, 'train.csv')
    test_file_path = os.path.join(raw_data_path, 'test.csv')
    train_df = pd.read_csv(train_file_path, index_col='PassengerId')
    test_df = pd.read_csv(test_file_path, index_col='PassengerId') 
    test_df['Survived'] = -888
    return pd.concat((train_df, test_df), sort=True)

def process_data(df):
    return (df
        .assign(Title = lambda df: df.Name.map(__get_title))
        .pipe(__fill_missing_values)
        .assign(FareBin = lambda df: pd.qcut(df.Fare, 4, labels=['low', 'medium', 'high', 'very_high']))
        .assign(AgeStatus = lambda df: np.where(df.Age >= 18, 'adult', 'child'))
        .assign(FamilySize = lambda df: df.Parch + df.SibSp + 1)
        .assign(IsMother = lambda df: np.where(((df.Sex=='female') & (df.Parch>0) & (df.AgeStatus=='Adult') & (df.Title != 'miss')), 1, 0))
        .assign(Cabin = lambda df: np.where(df.Cabin == 'T', np.nan, df.Cabin))
        .assign(Deck = lambda df: df.Cabin.map(__get_deck))
        .assign(IsMale = lambda df: df.Sex.map(lambda s: np.where(s == 'male', 1, 0)))
        .pipe(pd.get_dummies, columns=['Deck', 'Pclass', 'Title', 'FareBin', 'Embarked', 'AgeStatus'])
        .drop(['Cabin', 'Name', 'Ticket', 'Parch', 'SibSp', 'Sex'], axis=1)
        .pipe(__reorder_columns)
        )

def write_data(df):
    processed_data_path = os.path.join(os.path.pardir, 'data', 'processed')
    train_path = os.path.join(processed_data_path, 'train.csv')
    test_path = os.path.join(processed_data_path, 'test.csv')
    df.loc[df.Survived != -888].to_csv(train_path)
    test_columns = [col for col in df.columns if col != 'Survived']
    df.loc[df.Survived == -888][test_columns].to_csv(test_path)

def __get_title(name):
    first_name = name.split(',')[1]
    return first_name.split('.')[0].strip().lower() 

def __fill_missing_values(df):
    df.Embarked.fillna('C', inplace=True)

    median_fare = df.loc[(df.Embarked == 'S') & (df.Pclass == 3), 'Fare'].median()
    df.Fare.fillna(median_fare, inplace=True)

    title_age_median = df.groupby('Title').Age.transform('median')
    df.Age.fillna(title_age_median, inplace=True)

    return df

def __get_deck(cabin):
    if pd.isnull(cabin):
        return 'Z'
    else:
        return cabin[0].upper()

def __reorder_columns(df):
    columns = [column for column in df.columns if column != 'Survived']
    columns = ['Survived'] + columns
    return df[columns]

if __name__ == '__main__':
    df = read_data()
    df_processed = process_data(df)
    write_data(df_processed)

