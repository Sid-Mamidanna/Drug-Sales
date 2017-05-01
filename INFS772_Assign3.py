import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import Imputer

# this function takes the drugcount dataframe as input and output a tuple of 3 data frames: DrugCount_Y1,DrugCount_Y2,DrugCount_Y3
def process_DrugCount(drugcount):
    dc = pd.read_csv("DrugCount.csv")
    sub_map = {'1' : 1, '2':2, '3':3, '4':4, '5':5, '6':6, '7+' : 7}
    dc['DrugCount'] = dc.DrugCount.map(sub_map)
    dc['DrugCount'] = dc.DrugCount.astype(int)
    dc_grouped = dc.groupby(dc.Year, as_index=False)
    DrugCount_Y1 = dc_grouped.get_group('Y1')
    DrugCount_Y2 = dc_grouped.get_group('Y2')
    DrugCount_Y3 = dc_grouped.get_group('Y3')
    DrugCount_Y1.drop('Year', axis=1, inplace=True)
    DrugCount_Y2.drop('Year', axis=1, inplace=True)
    DrugCount_Y3.drop('Year', axis=1, inplace=True)
    return (DrugCount_Y1,DrugCount_Y2,DrugCount_Y3)

# this function converts strings such as "1- 2 month" to "1_2"
def replaceMonth(string):
    replace_map = {'0- 1 month' : "0_1", "1- 2 months": "1_2", "2- 3 months": "2_3", "3- 4 months": '3_4', "4- 5 months": "4_5", "5- 6 months": "5_6", "6- 7 months": "6_7", \
                   "7- 8 months" : "7_8", "8- 9 months": "8_9", "9-10 months": "9_10", "10-11 months": "10_11", "11-12 months": "11_12"}
    a_new_string = string.map(replace_map)
    return a_new_string

# this function processes a yearly drug count data
def process_yearly_DrugCount(aframe):
    processed_frame = None
    # aframe.drop("Year", axis = 1, inplace = True)
    reformed = aframe[['DSFS']].apply(replaceMonth)
    gd = pd.get_dummies(reformed)
    joined =  pd.concat([aframe, gd], axis = 1)
    joined.drop("DSFS", axis = 1, inplace = True)
    joined_grouped = joined.groupby("MemberID", as_index = False)
    processed_frame = joined_grouped.agg(np.sum)
    processed_frame.rename(columns = {'DrugCount' : 'Total_DrugCount'}, inplace = True)
    return processed_frame

# this is the function to split training dataset to training and test. You don't need to change the function
def split_train_test(arr, test_size=.3):
    train, test = train_test_split(arr, test_size=0.33)
    train_X = train[:, :-1]
    print train_X.shape
    train_y = train[:, -1]
    print train_y.shape
    test_X = test[:,:-1]
    test_y = test[:, -1]
    return (train_X,train_y,test_X,test_y)

# run linear regression. You don't need to change the function
def linear_regression((train_X,train_y,test_X,test_y)):
    regr = linear_model.LinearRegression()
    regr.fit(train_X, train_y)
    print 'Coefficients: \n', regr.coef_
    pred_y = regr.predict(test_X) # your predicted y values
    # The root mean square error
    mse = np.mean( (pred_y - test_y) ** 2)
    import math
    rmse = math.sqrt(mse)
    print ("RMSE: %.2f" % rmse)
    from sklearn.metrics import r2_score
    r2 = r2_score(pred_y, test_y)
    print ("R2 value: %.2f" % r2)

# for a real-valued variable, replace missing with median
def process_missing_numeric(Master_assn1, Total_DrugCount):
    # below is the code I used in the lecture ("exploratory_analysis.py") for dealing with missing values of the variable "age". You need to change the code below slightly
    variable_missing = np.where(Master_assn1['Total_DrugCount'].isnull(),1,0)
    medianTotalDrugCount = Master_assn1.Total_DrugCount.median()
    Master_assn1.Total_DrugCount.fillna(medianTotalDrugCount, inplace= True)
    Master_assn1.Total_DrugCount.hist(bins=20)
    plt.show()
    return (Master_assn1, Total_DrugCount)

# This function prints the ratio of missing values for each variable. You don't need to change the function
def print_missing_variables(df):
    for variable in df.columns.tolist():
        percent = float(sum(df[variable].isnull()))/len(df.index)
        print variable+":", percent

def main():
    pd.options.mode.chained_assignment = None # remove the warning messages regarding chained assignment. 
    daysinhospital = pd.read_csv('DaysInHospital_Y2.csv')
    drugcount = pd.read_csv('DrugCount.csv')
    li = map(process_yearly_DrugCount, process_DrugCount(drugcount))
    DrugCount_Y1_New = li[0]
    Master_Assn1 = pd.merge(daysinhospital, DrugCount_Y1_New)
    newColumns = ['MemberID', 'ClaimsTruncated', 'Total_DrugCount', 'DSFS_0_1', 'DSFS_1_2', 'DSFS_2_3', 'DSFS_3_4', 'DSFS_4_5', 'DSFS_5_6', 'DSFS_6_7', 'DSFS_7_8', 'DSFS_8_9', 'DSFS_9_10', 'DSFS_10_11', 'DSFS_11_12','DaysInHospital']
    Master_Assn1 = Master_Assn1[newColumns]
    print Master_Assn1.head(3)

    '''outputs:
     MemberID  ClaimsTruncated  Total_DrugCount  DSFS_0_1  DSFS_1_2  DSFS_2_3  \
0  24027423                0                3         0         0         0
1  98324177                0                1         1         0         0
2  33899367                1               23         1         0         1

   DSFS_3_4  DSFS_4_5  DSFS_5_6  DSFS_6_7  DSFS_7_8  DSFS_8_9  DSFS_9_10  \
0         1         0         0         0         0         0          0
1         0         0         0         0         0         0          0
2         1         1         1         1         1         1          1

   DSFS_10_11  DSFS_11_12  DaysInHospital
0           0           0               0
1           0           0               0
2           1           0               1
    '''
    ''' your code here for deal with missing values of the dummy variables. Please don't overthink this. You just need to write one line of code'''
    process_missing_numeric(Master_Assn1, 'Total_DrugCount')
    Master_Assn1.drop('MemberID', axis = 1, inplace=True)
    arr = Master_Assn1.values #Converts to 2d array
    linear_regression(split_train_test(arr))
    '''outputs:
    Coefficients:
[-0.05044987  0.01591138 -0.48598733 -0.15088138 -0.10255352 -0.08591492
 -0.080255   -0.07516292 -0.06195    -0.06551777 -0.06766854 -0.07096135
 -0.07170065 -0.07585565 -0.00627985]
RMSE: 0.26
R2 value: 0.57
    '''

if __name__ == '__main__':
    main()

