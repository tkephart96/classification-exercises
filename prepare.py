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
    # split
    train_validate, test = train_test_split(iris, test_size=.2, random_state=42, stratify=iris.species)
    train, validate = train_test_split(train_validate, 
                                        test_size=.25, 
                                        random_state=42, 
                                        stratify=train_validate.species)
    print('data cleaned, prepped, and split')
    return train, validate, test

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
    # split
    train_validate, test = train_test_split(titanic, test_size=.2, random_state=42, stratify=titanic.survived)
    train, validate = train_test_split(train_validate, 
                                        test_size=.25, 
                                        random_state=42, 
                                        stratify=train_validate.survived)
    print('data cleaned, prepped, and split')
    return train, validate, test

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
    # split
    train_validate, test = train_test_split(telco, test_size=.2, random_state=42, stratify=telco.churn)
    train, validate = train_test_split(train_validate, 
                                        test_size=.25, 
                                        random_state=42, 
                                        stratify=train_validate.churn)
    print('data cleaned, prepped, and split')
    return train, validate, test