# imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# functions
def prep_iris(df):
    """
    This function prepares the iris dataset by cleaning it, creating dummy variables for the species
    column, and splitting it into train, validate, and test sets.
    
    :param iris: The input dataset containing information about the iris flowers
    :return: The function `prep_iris` is returning three dataframes: `train`, `validate`, and `test`.
    These dataframes are the result of cleaning and splitting the original `iris` dataframe.
    """
    # clean
    df = df.drop(columns=['species_id','measurement_id'])
    df = df.rename(columns={'species_name':'species'})
    print('data cleaned and prepped')
    return df

def prep_titanic(df):
    """
    The function prepares the Titanic dataset by cleaning and splitting it into train, validate, and
    test sets.
    :return: The function `prep_titanic()` is returning three dataframes: `train`, `validate`, and
    `test`. These dataframes are the result of cleaning and splitting the original `titanic` dataframe.
    """
    # clean
    df = df.drop(columns=['age','class','deck','embark_town','passenger_id'])
    df['embarked'] = df.embarked.fillna(value='S')
    dummy_df = pd.get_dummies(df[['sex','embarked']], dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    print('data cleaned and prepped')
    return df

def prep_titanic_age(df):
    # clean
    df = df.drop(columns=['class','deck','embark_town','passenger_id'])
    df['embarked'] = df.embarked.fillna(value='S')
    dummy_df = pd.get_dummies(df[['sex','embarked']], dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    print('data cleaned and prepped')
    imputer = SimpleImputer(strategy = 'mean')
    df['age'] = imputer.fit_transform(df[['age']])
    return df

def prep_telco(df):
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
    df = df.drop(columns=['customer_id','payment_type_id','internet_service_type_id','contract_type_id'])
    df.loc[df.total_charges==' ','total_charges']=0
    df.total_charges = df.total_charges.astype(float)
    df['Female'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partnered'] = df.partner.map({'Yes': 1, 'No': 0})
    df['has_dependents'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['has_phone_service'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['does_paperless_billing'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churned'] = df.churn.map({'Yes': 1, 'No': 0})
    dummy_df = pd.get_dummies(df[['multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies','contract_type','internet_service_type','payment_type']], dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    print('data cleaned and prepped')
    return df

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
    st = [strat]
    train_validate, test = train_test_split(df, test_size=test, random_state=42, stratify=df[st])
    train, validate = train_test_split(train_validate, 
                                        test_size=validate, 
                                        random_state=42, 
                                        stratify=train_validate[st])
    print(f'train -> {train.shape}; {round(len(train)*100/len(df),2)}%')
    print(f'validate -> {validate.shape}; {round(len(validate)*100/len(df),2)}%')
    print(f'test -> {test.shape}; {round(len(test)*100/len(df),2)}%')
    return train, validate, test