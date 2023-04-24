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

def prep_split_iris(df, test=.2, validate=.25):
    """
    This function prepares and splits a given dataset into training, validation, and testing sets for
    machine learning purposes.
    
    :param df: The input dataframe containing the Iris dataset
    :return: three dataframes: train, validate, and test.
    """
    # clean
    df = df.drop(columns=['species_id','measurement_id'])
    df = df.rename(columns={'species_name':'species'})
    print('data cleaned and prepped')
    print('data split')
    train_validate, test = train_test_split(df, test_size=test, random_state=42, stratify=df['species'])
    train, validate = train_test_split(train_validate, 
                                        test_size=validate, 
                                        random_state=42, 
                                        stratify=train_validate['species'])
    print(f'train -> {train.shape}; {round(len(train)*100/len(df),2)}%')
    print(f'validate -> {validate.shape}; {round(len(validate)*100/len(df),2)}%')
    print(f'test -> {test.shape}; {round(len(test)*100/len(df),2)}%')
    return train, validate, test

def prep_titanic(df):
    """
    The function preps a Titanic dataset by dropping certain columns, filling missing values, creating
    dummy variables, and returning the cleaned dataset.
    
    :param df: a pandas DataFrame containing the Titanic dataset with columns for age, class, deck,
    embark_town, passenger_id, sex, embarked, and other variables. The function `prep_titanic` takes
    this DataFrame as input and performs some data cleaning and preparation steps on it
    :return: a cleaned and prepped dataframe.
    """
    # clean
    df = df.drop(columns=['age','class','deck','embark_town','passenger_id'])
    df['embarked'] = df.embarked.fillna(value='S')
    dummy_df = pd.get_dummies(df[['sex','embarked']], dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    print('data cleaned and prepped')
    return df

def prep_titanic_age(df):
    """
    This function prepares the Titanic dataset by cleaning and prepping the data, including dropping
    unnecessary columns, filling missing values, creating dummy variables, and imputing missing age
    values with the mean.
    
    :param df: The input dataframe that contains information about passengers on the Titanic
    :return: a cleaned and prepped dataframe with the 'age' column imputed using the mean strategy.
    """
    # clean
    df = df.drop(columns=['class','deck','embark_town','passenger_id'])
    df['embarked'] = df.embarked.fillna(value='S')
    dummy_df = pd.get_dummies(df[['sex','embarked']], dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    print('data cleaned and prepped')
    imputer = SimpleImputer(strategy = 'mean')
    df['age'] = imputer.fit_transform(df[['age']])
    return df

def prep_split_titanic(df, test=.2, validate=.25):
    """
    This function prepares and splits Titanic dataset into train, validate, and test sets.
    
    :param df: The input dataframe containing the Titanic dataset
    :return: three dataframes: train, validate, and test.
    """
    # clean
    df = df.drop(columns=['age','class','deck','embark_town','passenger_id'])
    df['embarked'] = df.embarked.fillna(value='S')
    dummy_df = pd.get_dummies(df[['sex','embarked']], dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    print('data cleaned and prepped')
    print('data split')
    train_validate, test = train_test_split(df, test_size=test, random_state=42, stratify=df['survived'])
    train, validate = train_test_split(train_validate, 
                                        test_size=validate, 
                                        random_state=42, 
                                        stratify=train_validate['survived'])
    print(f'train -> {train.shape}; {round(len(train)*100/len(df),2)}%')
    print(f'validate -> {validate.shape}; {round(len(validate)*100/len(df),2)}%')
    print(f'test -> {test.shape}; {round(len(test)*100/len(df),2)}%')
    return train, validate, test

def prep_telco(df):
    """
    The function takes a dataframe and performs data cleaning and preparation by dropping unnecessary
    columns, converting data types, creating dummy variables, and mapping categorical variables to
    binary values.
    
    :param df: a pandas DataFrame containing Telco customer data
    :return: a cleaned and preprocessed dataframe.
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

def prep_split_telco(df, test=.2, validate=.25):
    """
    This function prepares and splits a Telco customer dataset into training, validation, and testing
    sets.
    
    :param df: The input dataframe that contains the Telco customer data
    :return: three dataframes: train, validate, and test.
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
    print('data split')
    train_validate, test = train_test_split(df, test_size=test, random_state=42, stratify=df['churn'])
    train, validate = train_test_split(train_validate, 
                                        test_size=validate, 
                                        random_state=42, 
                                        stratify=train_validate['churn'])
    print(f'train -> {train.shape}; {round(len(train)*100/len(df),2)}%')
    print(f'validate -> {validate.shape}; {round(len(validate)*100/len(df),2)}%')
    print(f'test -> {test.shape}; {round(len(test)*100/len(df),2)}%')
    return train, validate, test

def split_data(df, strat, test=.2, validate=.25):
    """
    This function splits a given dataframe into training, validation, and test sets based on a given
    stratification column and specified proportions.
    
    :param df: The input dataframe that needs to be split into train, validate, and test sets
    :param strat: The strat parameter is the name of the column in the dataframe that will be used for
    stratified sampling. Stratified sampling is a technique used to ensure that the distribution of a
    certain variable in the dataset is maintained in the training, validation, and test sets
    :param test: The proportion of the data that should be allocated to the test set. In this case, it
    is set to 0.2 or 20% of the data
    :param validate: The "validate" parameter is the proportion of the data that will be used for
    validation. It is set to 0.25, which means that 25% of the data will be used for validation
    :return: The function `split_data` returns three dataframes: `train`, `validate`, and `test`.
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