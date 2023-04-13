import pandas as pd
import numpy as np
import os
from env import get_db_url

def get_titanic_data():  # sourcery skip: remove-unnecessary-else
    filename = "titanic.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('select * from passengers', get_db_url('titanic_db'))
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)
        # Return the dataframe to the calling code
        return df

def get_iris_data():  # sourcery skip: remove-unnecessary-else
    filename = "iris.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('select * from species join measurements using (species_id)', get_db_url('iris_db'))
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)
        # Return the dataframe to the calling code
        return df

def get_telco_data():  # sourcery skip: remove-unnecessary-else
    filename = "telco.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('select * from customers join contract_types using(contract_type_id) join internet_service_types using(internet_service_type_id) join payment_types using(payment_type_id)', get_db_url('telco_churn'))
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)
        # Return the dataframe to the calling code
        return df
