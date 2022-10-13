import sys
import pandas as pd
import string
import pickle
import re
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    Loads the data from the sql database in database_filepath
    return feature as X and results as y

    Args:
        database_filepath: path to the sql database

    returns:
        X: panda dataframe : feature: list of messages
        y: panda dataframe : result: classification of each message
        category_names: list of categories
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_messages', engine)

    # As feature is used the messages (translated in english)
    X = df['message']
    # classification for each message
    y = df.iloc[:, 3:]

    return X, y, y.columns


def tokenize(text, stop_words=set(stopwords.words('english'))):
    ''' Remove punctuation, strip unncessary spaces, lematize,
    remove stopwords and tokenize text

    Args:
        text: str
    returns:
        clean_tokens: list of tokens
    '''
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[0-9]+', '', text)
    tokens = word_tokenize(text)

    filtered_tokens = [w for w in tokens if not w.lower() in stop_words]

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in filtered_tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    '''
    build the model

    Args: None

    returns:
        pipeline
    '''
    pipeline = Pipeline([
                    ('vect',
                    CountVectorizer(tokenizer=tokenize, ngram_range= (1, 2))),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(
                    RandomForestClassifier(min_samples_split=2, n_estimators=20)))
                    ])

    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    ''' calculate the f1 score for each category and overall average
    Args:
        model: trained model
        X_test: test data
        y_test: expected output for X_test
        category_names: list of categories / features
    Return:
        print f1 score for categories and average over all categories
        df_results: pandas dataframe with all results
    '''
    # make model predictions
    y_pred = model.predict(X_test)

    # Creating a dictionary to store results
    results = {}

    # looping though all categories an storing their precision, recall and f1_score
    for i in range(0, 36):
        precision, recall, fscore, support = score(y_test.iloc[:,i],y_pred[:,i],average='weighted')
        # adding results to the dictionary
        results[category_names[i]] = [precision, recall, fscore]

    # Creating pandas dataframe with results
    df_results = pd.DataFrame.from_dict(results, orient='index', columns=['precision', 'recall', 'fscore'])

    # printing overall modell performance
    print('overall f1_score: {}\n'.format(df_results.fscore.mean()))

    # printing category results
    print(df_results)

    return df_results


def save_model(model, model_filepath):
    ''' save the model to disk
    Arg:
        model: Machine learning model
        model_filepath: str path to file
    returns: None
    '''
    # save the model to dish to model_filepath
    pickle.dump(model, open(model_filepath, 'wb'))

    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
