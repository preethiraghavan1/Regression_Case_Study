import pandas as pd
from zipfile import ZipFile
import numpy as np
import math as m
import pandas as pd

#Location of data folder in your local machine
# WIll change later - Preethi
data_file_location = "/Users/predev/Documents/Learn/MachineLearning/Galvanize/Assessments/Data/Train.zip"
data_file_csv_location = "/Users/predev/Documents/Learn/MachineLearning/Galvanize/Assessments/Data/Train.csv"
data_file_csv_Mode_location = "/Users/predev/Documents/Learn/MachineLearning/Galvanize/Assessments/Data/Train_clean_Mode.csv"

# Start
#original line - to be uncommented in the end - Preethi
#zf = ZipFile('../data/Train.zip')

#original line - to be commented in the end - Preethi
#zf = ZipFile('../data/Train.zip')
zf = ZipFile(data_file_location)

def delete_rows_with_value(df, value = [None]) :
    df = df[~df.isin(value).any(axis = 1)]
    return df

def delete_columns_with_value(df, value = [None]) :
    df = df[df.columns[~df.isin(value).any(axis = 0)]]
    return df

##########################

from enum import Enum
class Replace(Enum):
     MEAN = 1
     MODE = 2
     VALUE = 3

#dependent : enum Replace
def replace_given_values(df, replaceVal, value = [None]) :
    df1 = df.replace(value, np.nan)
    if(type(replaceVal) == Replace) :
        if(replaceVal == Replace.MEAN) :
            df1 = df1.fillna(df1.mean(skipna = True, axis = 0))
        elif(replaceVal == Replace.MODE) :
            df1 = df1.fillna(df1.mode(axis = 0).ix[0])
    elif (type(replaceVal) in (tuple, list)): #Problem with python :( one extra tab this becomes the else for another (Wrong) if condition
        df1 = df1.fillna(dict(enumerate(replaceVal)))
    else : #Problem with python :( one extra tab this becomes the else for another (Wrong) if condition
        df1 = df1.fillna(replaceVal)
    return df1

#dependent : enum Replace
#dependent : replace_given_values
def replace_given_values_With_Mean(df, value = [None]) :
    return replace_given_values(df, Replace.MEAN, value=value)

#dependent : enum Replace
#dependent : replace_given_values
def replace_given_values_With_Mode(df, value = [None]) :
    return replace_given_values(df, Replace.MODE, value=value)

##########################

def map_UsageBand_to_numbers(x) :
    dic = {'Low' : 1, "Medium" : 2, "High" : 3}
    return dic[x] if x in dic else np.NaN

def map_productSize_to_numbers(x) :
    dic = {"Mini" : 1, 'Small' : 2, "Compact" : 3, "Medium" : 4, 'Large / Medium' : 5, 'Large':6}
    return dic[x] if x in dic else np.NaN

def getNumbers(x) :
    try :
     return float(x)
    except :
        return None
def split_val(x) :
    x = x.replace(",","-").replace(" to","to").replace("to ","to").replace(" -","-").replace("- ","-")
#     print x
    sp = x.split("-")
    val = sp[-1]
#     print sp
#     tot_rows =
    num_str = [q for q in sp[-1].split() if "to" in q]
    if(len(num_str) > 0) :
        mean_val =  np.mean(map(getNumbers, num_str[0].split("to")))
    else :
        mean_val = np.NaN
    return pd.Series([mean_val, val.split()[-1]])
def inchesToNumber1(x) :
    return inchesToNumber(x, True)
def inchesToNumber(x, foot=False) :
    if(x == 'nan') | (x == np.NaN) :
        return np.NaN
    if(x == 'None or Unspecified') :
        return x
    if(getNumbers(x)) :
        return float(x)
    numbers = [getNumbers(q.replace("'","").replace('"',"")) for q in x.split() if ("inch" not in q)]
    if(foot == True) :
        if(numbers[0]) :
           return numbers[0]
        else :
            return x
    else :
        if(len(numbers) == 1) :
           inch = numbers[0]
           return float(inch)/12
        if(len(numbers) == 2) :
           feet = numbers[1]
           inch = numbers[0] if (numbers[0]) else None
    return feet + float(inch)/12

def clean_replace_data_with_mode(df) :
    empty_values = [np.NaN,'nan',  None, 'None or Unspecified']
    df1 = replace_given_values_With_Mode(df, empty_values)
    df1.to_csv(data_file_csv_Mode_location)
    return df1
    

def initial_cleaning(df):
    #Start Cleaning - Preethi
    #apply - low = 1, Medium = 2, High = 3
    df['UsageBand'] = df['UsageBand'].apply(map_UsageBand_to_numbers)
    df['ProductSize'] = df['ProductSize'].apply(map_productSize_to_numbers)

    #numbers in inhes and feet to just decimal feet
    df['Undercarriage_Pad_Width'] = df['Undercarriage_Pad_Width'].apply(inchesToNumber)
    df['Stick_Length'] = df['Stick_Length'].apply(inchesToNumber)
    df['Blade_Width'] = df['Blade_Width'].apply(inchesToNumber1)
    df['Tire_Size'] = df['Tire_Size'].apply(inchesToNumber1)

    #fiProductClassDesc column to multiple type columns
    ser = df['fiProductClassDesc']
    ser = ser.apply(split_val)
    newdf = pd.get_dummies(ser[1])
    for col in newdf.columns:
       newdf[col] =  newdf[col] * ser[0]

    df = pd.merge(df, newdf, left_index = True, right_index = True)
    df = df.drop('fiProductClassDesc', axis = 1)
    #End Cleaning - Preethi

df = pd.read_csv(data_file_csv_location)
dforig = df.copy()

df = clean_replace_data_with_mode(df)
clean_replace_data_with_mode(df)

#End - Preethi


# def delete_rows_with_empty_values(df, values) :




#delete rows which are empty



df = pd.read_csv(data_file_location)

year = df['YearMade']
year = year[year != 1000]

price_v_year = df[['SalePrice', 'YearMade']]
