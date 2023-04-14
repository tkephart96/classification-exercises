# imports
import pandas as pd
import os
from env import get_db_url

# functions
def get_titanic_data():
    """
    This function checks if a CSV file exists, reads it if it does, and if not, reads data from a SQL
    database, saves it to a CSV file, and returns the data.
    :return: The function `get_titanic_data()` returns a pandas DataFrame containing the Titanic
    passenger data. If the data has been previously cached as a CSV file, it reads the data from the
    file. Otherwise, it reads the data from a SQL database, caches it as a CSV file, and returns the
    DataFrame.
    """
    filename = "titanic.csv"
    if os.path.isfile(filename):
        print('csv file found and loaded')
        return pd.read_csv(filename)
    else:
        print('creating df and exporting csv')
        # read the SQL query into a dataframe
        df = pd.read_sql('select * from passengers', get_db_url('titanic_db'))
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        # Return the dataframe to the calling code
        return df

def get_iris_data():
    """
    This function checks if a CSV file exists, and if it does, it returns the data from the file,
    otherwise it reads data from a SQL database, saves it to a CSV file, and returns the data.
    :return: The function `get_iris_data()` returns a pandas DataFrame containing the iris data. If the
    data is already cached in a CSV file named "iris.csv", it reads the data from the file. Otherwise,
    it reads the data from a SQL database named "iris_db", joins the "species" and "measurements"
    tables, caches the data in a CSV file, and returns the DataFrame.
    """
    filename = "iris.csv"
    if os.path.isfile(filename):
        print('csv file found and loaded')
        return pd.read_csv(filename)
    else:
        print('creating df and exporting csv')
        # read the SQL query into a dataframe
        df = pd.read_sql('select * from species join measurements using (species_id)', get_db_url('iris_db'))
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        # Return the dataframe to the calling code
        return df

def get_telco_data():
    """
    This function reads telco data from a CSV file if it exists, otherwise it reads the data from a SQL
    database and saves it to the CSV file for future use.
    :return: The function `get_telco_data()` returns a pandas DataFrame containing data from either a
    CSV file named "telco.csv" or a SQL query from a database named "telco_churn". If the CSV file
    exists, it reads the data from the file, otherwise it reads the data from the SQL query, saves it to
    the CSV file for caching, and returns the DataFrame.
    """
    filename = "telco.csv"
    if os.path.isfile(filename):
        print('csv file found and loaded')
        return pd.read_csv(filename)
    else:
        print('creating df and exporting csv')
        # read the SQL query into a dataframe
        df = pd.read_sql('select * from customers join contract_types using(contract_type_id) join internet_service_types using(internet_service_type_id) join payment_types using(payment_type_id)', get_db_url('telco_churn'))
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        # Return the dataframe to the calling code
        return df
