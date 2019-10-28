import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix



def counting_uniques(df):
    '''
    For the given dataframe, gives a sum of the unique values in each feature.
    Then prints a plot bar to represent that.
    '''
    print (df.nunique())
    print (df.nunique().plot.bar())


# Pre-Process Data
def Detect_missing_value(df):
    '''
    Find out the columns have missing values
    Input:
        pandas dataframe
 
    Returns:
        a list of those column names
    '''
    rv = []
    for col in df.columns:
        if df[col].count() < df.shape[0]:
            rv.append(col)
            print(col, "has missing values.")
    return rv


def fill_missing(df, cols,  method="mean"):

    for col in cols:
        if df[col].dtype == 'object':
            df[col].fillna("Missing", inplace=True)
                
        elif df[col].dtype == 'int' or df[col].dtype == 'float':
            if method == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            if method == "median":
                df[col].fillna(df[col].median(), inplace=True)
        print ('Filling missing value for {} using {}'.format(col, method))
    return df


def Separate_Col(df):
    '''
    Separting the numeric columns and categorical columns

    Inputs:
        pandas dataframe
 
    Returns:
        two list of categorical column and numeric column names
    '''
    strcol = []
    numcol =[]
    for col in df.columns:
        if df[col].dtype == 'O':
            strcol.append(col)
        elif df[col].dtype != '<M8[ns]':
            numcol.append(col)
            
    return strcol, numcol


def Convert_datetime(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col])
    return df


# CREATE DUMMIES

def dummy_variable(variable, df):
    '''
    Using the binned columns, replace them with dummy columns.
    Inputs:
    df: A panda dataframe
    variable: A list of column headings for binned variables
    Outputs:
    df:A panda dataframe
    '''
    dummy_df = pd.get_dummies(df[variable]).rename(columns = lambda x: str(variable)+ str(x))
    df = pd.concat([df, dummy_df], axis=1)
    df.drop([variable], inplace = True, axis=1)
    
    return df

def bin_gen(df, variable, label, fix_value):
    '''
    Create a bin column for a given variable, derived by using the 
    description of the column to determine the min, 25, 50, 75 and max
    of the column. Then categorize each value in the original variable's
    column in the new column, labeled binned_<variable>, with 1,2,3,4
    Ranging from min to max
    Inputs:
    df: A panda dataframe
    variable: A string, which is a column in df
    label: A string
    fix_value: Either prefix or suffix
    Outputs:
    df: A panda dataframe
    '''
    variable_min = df[variable].min()
    variable_25 = df[variable].quantile(q = 0.25)
    variable_50 = df[variable].quantile(q = 0.50)
    variable_75 = df[variable].quantile(q = 0.75)
    variable_max = df[variable].max()
    
    bin = [variable_min, variable_25, variable_50, variable_75, variable_max]
    unique_values = len(set(bin))
    
    label_list = []
    iterator = 0
    for x in range(1, unique_values):
        iterator += 1
        label_list.append(iterator)
    
    if fix_value == 'prefix':
        bin_label = label + variable
    elif fix_value == 'suffix':
        bin_label = variable + label
    
    df[bin_label] = pd.cut(df[variable], bins = bin, include_lowest = True, labels = label_list)
    df.drop([variable], inplace = True, axis=1)
    
    df = dummy_variable(bin_label, df)
    
    return df


# Visualize
def corr_matrix(df):
    '''
    Creates a heatmap that shows the correlations between the different variables in a dataframe.
    
    Input:
        df: a dataframe
        title: name of the correlation_matrix
        
    Return:
        Outputs a heatmatrix showing correlations
    
    
    '''
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


def plotting_curve (dataframe, column, title):
    '''
    Given a dataframe, a column name, and a title,
        displays a plot of that dataframe column distribution.
        
    Input:
        dataframe
        column: column name (string)
        title: string
        
    Return:
        displays a distribution of that variable
        
    Inspired by:
        https://seaborn.pydata.org/generated/seaborn.distplot.html
    '''
    try:
        ax = sns.distplot(dataframe[column])
        ax.set_title(title)
        plt.show()
    except:
        pass


# Classifiers

def split_data(df, depv):
    '''
    Split the data into training and testing set
    
    And save them to run try different models
    '''
    y = df[depv]
    indepv = list(df.columns)
    indepv.remove('loan_status')
    x = df[indepv]
    # get train/test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=1234)
    
    return x_train, x_test, y_train, y_test


def Classifier(model, num, x_train, y_train, x_test):
    if model == 'LR':
        clf = LogisticRegression('l2', C=num)
    elif model == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=num)
    elif model == 'DT':
        clf = DecisionTreeClassifier(max_depth=num)
    elif model == 'RF':
        clf = RandomForestClassifier(max_depth=num)
    elif model == 'BAG':
        clf = BaggingClassifier(max_samples=num, bootstrap=True, random_state=0)
    elif model == 'BOOST':
        clf = GradientBoostingClassifier(max_depth=num)
    
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred
