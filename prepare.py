# imports
import pandas as pd
from sklearn.model_selection import train_test_split

# functions
def prep_iris(iris):
    """
    This function prepares the iris dataset by cleaning it, creating dummy variables for the species
    column, and splitting it into train, validate, and test sets.
    
    :param iris: The input dataset containing information about the iris flowers
    :return: The function `prep_iris` is returning three dataframes: `train`, `validate`, and `test`.
    These dataframes are the result of cleaning and splitting the original `iris` dataframe.
    """
    # clean
    iris = iris.drop_duplicates()
    iris = iris.drop(columns=['species_id','measurement_id'])
    iris = iris.rename(columns={'species_name':'species'})
    dummy_iris = pd.get_dummies(iris.species, drop_first=True)
    iris = pd.concat([iris, dummy_iris], axis=1)
    print('data cleaned and prepped')
    return iris

def prep_titanic(titanic):
    """
    The function prepares the Titanic dataset by cleaning and splitting it into train, validate, and
    test sets.
    :return: The function `prep_titanic()` is returning three dataframes: `train`, `validate`, and
    `test`. These dataframes are the result of cleaning and splitting the original `titanic` dataframe.
    """
    # clean
    titanic = titanic.drop_duplicates()
    titanic = titanic.drop(columns=['age','class','deck','embark_town'])
    titanic['embarked'] = titanic.embarked.fillna(value='S')
    dummy_titanic = pd.get_dummies(titanic[['sex','embarked']], drop_first=True)
    titanic = pd.concat([titanic, dummy_titanic], axis=1)
    print('data cleaned and prepped')
    return titanic

def prep_telco(telco):
    """
    The function takes a telco dataset, cleans it, creates dummy variables for categorical columns, and
    splits it into train, validate, and test sets.
    
    :param telco: `telco` is a pandas DataFrame containing data related to a telecommunications
    company's customers, such as their demographic information, services subscribed to, and payment
    information
    :return: The function `prep_telco` is returning three dataframes: `train`, `validate`, and `test`.
    These dataframes are the result of splitting and cleaning the original `telco` dataframe.
    """
    # clean
    telco = telco.drop_duplicates()
    telco = telco.drop(columns=['customer_id','payment_type_id','internet_service_type_id','contract_type_id'])
    telco.total_charges[telco.total_charges==' ']=0
    telco.total_charges = telco.total_charges.astype(float)
    telco_obj = telco.select_dtypes(include='object').columns.to_list()
    dummy_telco = pd.get_dummies(telco[telco_obj], drop_first=True)
    telco = pd.concat([telco, dummy_telco], axis=1)
    print('data cleaned and prepped')
    return telco

def split_data(df, strat, test=.2, validate=.25):
    """
    The function splits a given dataframe into training, validation, and test sets based on a specified
    stratification variable.
    
    :param df: The input dataframe that needs to be split into train, validate, and test sets
    :param strat: The name of the column in the dataframe that will be used for stratified sampling
    during the data split
    :param test: The proportion of the data that should be allocated to the test set. In this case, it
    is set to 0.2 or 20% of the data
    :param validate: The proportion of the data that will be used for validation. It is set to 0.25,
    which means that 25% of the data will be used for validation
    :return: three dataframes: train, validate, and test.
    """
    print('data split')
    train_validate, test = train_test_split(df, test_size=test, random_state=42, stratify=df[{strat}])
    train, validate = train_test_split(train_validate, 
                                        test_size=validate, 
                                        random_state=42, 
                                        stratify=train_validate[{strat}])
    print(f'train -> {train.shape}; {round(len(train)*100/len(df),2)}%')
    print(f'validate -> {validate.shape}; {round(len(validate)*100/len(df),2)}%')
    print(f'test -> {test.shape}; {round(len(test)*100/len(df),2)}%')
    return train, validate, test