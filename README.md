# Disaster Response Pipeline Project

This project uses messages from disaster and classifies them into one of the 36 provided categories. This classification result can be used to send the message to the adequate disaster response team.

The project start with gathering and cleaning data from a csv file and stores into a database.
A classifier model is trained on the clean data.
An app has 2 main section:
- top section allows the user to type a message and get the corresponding classification using the trained model created in the previous step.
- bottom section shows 3 visualisations for the dataset used for training: genre, histogram of categories and correlation matrix between each categories.   

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/
