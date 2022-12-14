import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' load and merge both csv files into a panda dataframe
    Args:
        messages_filepath: str Path the the messages csv file
        categories_filepath: str Path the the categories csv file

    returns
        df: dataframe
    '''
        # load messages dataset
    messages = pd.read_csv(messages_filepath, index_col='id')
    # load categories dataset
    categories = pd.read_csv(categories_filepath, index_col='id')

    # merge datasets only keeping identical ids
    df = messages.merge(categories, on='id', how='inner')

    return df


def clean_data(df):
    ''' Clean the categories column into a one-hot encoding format.
    Separate data and column labels from each other.

    Args:
        df: Dataframe

    returns
        result: dataframe with labelled columns and one-hot encoding
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories.
    # use the text before the "-" to name columns
    category_colnames = row.str.split('-', expand=True).iloc[:,0].values

    # rename the columns of `categories`
    categories.columns = category_colnames

    # extract the values for one hot encoding and cast it as int
    # The value is the last character of each cell
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Replace values "2" to "1" in column "related"
    # assuming 2 is "active" and same meaning than "1"
    categories['related'].replace(2, 1, inplace=True)

    # drop the original categories column from current dataframe
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the cleaned categories dataframe
    df = df.merge(categories, on='id', how='left')

    # remove duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''
    Saves the dataframe df in an sql database

    Args:
        df: dataframe
        database_filename: path to the file, where the database should be saved

    returns: None
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')
    pass


def main():
    '''
    Load clean and saves data
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
