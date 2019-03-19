# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:19:38 2019

@author: 16514
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import os
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import re
from sklearn.covariance import EllipticEnvelope

mppt1 = [1, 2, 3]
mppt2 = [5, 6, 7]
mppt3 = [9, 10, 11, 12]
path = 'E:/work/data/solar/solar/相关信息/上海西门子.xlsx'
table = pd.read_excel(path)

# 输入参数关断器，返回其属于的组串
def find_substring_num(dc_devId):
    substring_num = table[table['dc_devId'] == int(dc_devId)]['substring_num']
    return substring_num


def find_string_inverter_name(dc_devId):
    substring_num = table[table['dc_devId'] == int(dc_devId)]['string_inverter_name']
    return substring_num


# 解析文件名，得到逆变器，关断器，日期
def parse_file_name(file_name):
    date = re.findall('(.{4}-.{2}-.{2})', file_name)[0]
    interver_index = re.findall(' ..... ', file_name)[0]
    shutoff_index = re.findall('[0-9]{12}', file_name)[0]
    return date, interver_index, shutoff_index


# 分析数据
# 1.outlier
def calcu(mppt):
    clf = IsolationForest(random_state=3, contamination=0.03)
    #mpptrest=mppt[['关断器', '子串号', '日期', '组件', '逆变器']]
    #mppt_test = mppt[mppt.columns.difference(['关断器', '子串号', '日期', '组件', '逆变器'])].iloc[:,:-3]
    mpptrest=mppt[['日期', '组件']]
    mppt_test = mppt[mppt.columns.difference(['日期', '组件'])]
    # fig=plt.figure()
    # ax = fig.add_subplot()
    # mppt_test.plot(ax=ax)
    # plt.show()
    try:
        clf.fit(mppt_test)
    except ValueError:
        return None

    y_pred = clf.predict(mppt_test)
    rowmean=mppt_test.values.mean(axis=1)
    allmean=mppt_test.values.mean()
    output = mpptrest[(y_pred == -1) & (rowmean<allmean)]
    output=output[['日期', '组件']]
    #np.array(output['关断器']).astype(str)
    print(output)
    print('======')
    return output


# 2.regressor
def calcu2(mppt):
    clf = EllipticEnvelope(contamination=0.01)

    my_mppt1 = mppt.iloc[:, 0:106]
    clf.fit(my_mppt1)
    y_pred = clf.predict(my_mppt1)
    # y_pred = clf.predict(my_mppt1)
    output = mppt[y_pred == -1].iloc[:, 108]
    return output


# 3.formula
def calcu3(mppt):
    clf = LocalOutlierFactor(contamination=0.4)
    my_mppt1 = mppt.iloc[:, 0:106]
    clf.fit(my_mppt1)
    y_pred = clf.fit_predict(my_mppt1)
    output = mppt[y_pred == -1].iloc[:, 108]
    return output


def parse_file(file_name='E:/work/data/solar/solar/optimizer_data/optimizer_201902_19.csv'):
    output_all=pd.DataFrame()
    df = pd.read_csv(file_name)
    if df.shape[0]<20:
        return pd.DataFrame()
    # 时间列表
    date_unique = df['REV1'].unique()
    opt_unique = df['OPT_NO'].unique()
    channel_unique = df['CHANNEL'].unique()
    for date in date_unique:
        datamppt1 = pd.DataFrame()
        for opt in opt_unique:
            for channel in channel_unique:
                zujian = df[(df[u'REV1'] == date) & (df[u'OPT_NO'] == opt) & (df[u'CHANNEL'] == channel)]
                substring_num = find_substring_num(opt)
                string_inverter_name = find_string_inverter_name(opt)
                new = zujian['INPUT_POWER']
                new = new.reset_index(drop=True)
                if len(list(string_inverter_name))==0:
                    continue
                else:
                    new['逆变器'] = list(string_inverter_name)[0]
                new['日期'] = date
                new['关断器'] = opt
                if len(list(substring_num)) > 0:
                    new['子串号'] = list(substring_num)[0]
                new['组件'] = channel
                # datamppt1=pd.concat([datamppt1,new],axis=1,sort=False)
                datamppt1 = datamppt1.append(new)

        output = calcu(datamppt1)
        output_all=pd.concat([output_all,output])
    return output_all


# and (df[u'OPT_NO'] ==opt) and (df[u'CHANNEL']==channel)


# 传入文件夹
# class shutoff():
#     def __init__(self, file_name):
#         datamppt1 = pd.DataFrame()
#         datamppt2 = pd.DataFrame()
#         datamppt3 = pd.DataFrame()
#         for root, dirs, files in os.walk(file_name):
#             for file in files:
#                 date, interver_index, shutoff_index = parse_file_name(file)
#                 substring_num = find_substring_num(shutoff_index)
#                 file = ''.join(['C:/Users/lenovo/Desktop/solar/data_shanghai_guanduanqi/', file])
#                 shutoff_data = pd.read_excel(file)
#                 aa = int(substring_num)
#
#             self.mppt1 = datamppt1
#             self.mppt2 = datamppt2
#             self.mppt3 = datamppt3
#
#     def result(self):
#         mppt1 = calcu(self.mppt1)
#         mppt2 = calcu(self.mppt2)
#         mppt3 = calcu(self.mppt3)
#         mppt1_2 = calcu2(self.mppt1)
#         mppt2_2 = calcu2(self.mppt2)
#         mppt3_2 = calcu2(self.mppt3)
#         mppt1_3 = calcu3(self.mppt1)
#         mppt2_3 = calcu3(self.mppt2)
#         mppt3_3 = calcu3(self.mppt3)
#         self.outlier = pd.concat([mppt1, mppt2, mppt3])
#         self.outlier_2 = pd.concat([mppt1_2, mppt2_2, mppt3_2])
#         self.outlier_3 = pd.concat([mppt1_3, mppt2_3, mppt3_3])
#
#     def to_excel(self):
#         outlier_path = 'C:/Users/lenovo/Desktop/solar/shanghai_guanduanqi_result/outlier.xlsx'
#         regressor_path = 'C:/Users/lenovo/Desktop/solar/shanghai_guanduanqi_result/regressor.xlsx'
#         formula_path = 'C:/Users/lenovo/Desktop/solar/shanghai_guanduanqi_result/formula.xlsx'
#         intersection_path = 'C:/Users/lenovo/Desktop/solar/shanghai_guanduanqi_result/intersection.xlsx'
#         outlier = self.outlier
#         outlier_2 = self.outlier_2
#         outlier_3 = self.outlier_3
#         intersection = pd.Series(list(set(outlier).intersection(set(outlier_2))))
#         if not os.path.exists(outlier_path):
#             outlier.to_excel(outlier_path)
#         if not os.path.exists(regressor_path):
#             outlier_2.to_excel(regressor_path)
#         if not os.path.exists(intersection_path):
#             intersection.to_excel(intersection_path)
#         if not os.path.exists(formula_path):
#             outlier_3.to_excel(formula_path)

def to_csv(output_all,file_name):
    if not os.path.exists(file_name):
        output_all.to_csv(file_name)

def walk(rootdir):
    g = os.walk(rootdir)
    for path, dir_list, file_list in g:
        return path,file_list



if __name__ == '__main__':
    path,file_names=walk('..\\optimizer_data')
    for i, file in enumerate(file_names):
        if i>5:
            file=''.join([path,'\\',file])
            output_all=parse_file(file)

            file_name=''.join(['..\\output\\output',str(i),'.csv'])
            to_csv(output_all,file_name)
            

file_name='E:/work/data/solar/solar/optimizer_data/optimizer_201902_18.csv'
output_all=pd.DataFrame()
df = pd.read_csv(file_name)
date_unique = df['REV1'].unique()
opt_unique = df['OPT_NO'].unique()
channel_unique = df['CHANNEL'].unique()
ns5min=5*60*1000000000
df['ECU_SEND_LOCALTIME']=df['ECU_SEND_LOCALTIME'].apply(lambda x:np.datetime64(x).astype('datetime64[m]'))
df['ECU_SEND_LOCALTIME']=pd.to_datetime(((df['ECU_SEND_LOCALTIME'].astype(np.int64)// ns5min) *ns5min))
df['exactdate']=df['ECU_SEND_LOCALTIME'].apply(lambda x:np.datetime64(x).astype('datetime64[D]'))
#(df['ECU_SEND_LOCALTIME']-df['exactdate']).astype('timedelta64[m]').unique()
df=df[((df['ECU_SEND_LOCALTIME']-df['exactdate']).astype('timedelta64[m]')>=540) & ((df['ECU_SEND_LOCALTIME']-df['exactdate']).astype('timedelta64[m]')<=900)]

#date_unique = df['REV1'].unique()
#opt_unique = df['OPT_NO'].unique()
#channel_unique = df['CHANNEL'].unique()
#i=1
#j=10
#channel=1
#date=date_unique[i]
#zujian = df[((df['REV1'] == date_unique[0])) & ((df['ECU_SEND_LOCALTIME']-df['exactdate']).astype('timedelta64[m]')>=540) & ((df['ECU_SEND_LOCALTIME']-df['exactdate']).astype('timedelta64[m]')<=570)]
#zujian.columns
#new = zujian[['ECU_SEND_LOCALTIME','OPT_NO','CHANNEL','INPUT_POWER']]
#new = new.reset_index(drop=True)


for date in date_unique:     
           #substring_num = find_substring_num(opt)
           #string_inverter_name = find_string_inverter_name(opt)      
        for j in range(9,15,1):
            datamppt1 = pd.DataFrame()
            for channel in channel_unique:
                zujian = df[(df[u'REV1'] == date) & (df[u'CHANNEL'] == channel) & ((df['ECU_SEND_LOCALTIME']-df['exactdate']).astype('timedelta64[m]')>60*j) & ((df['ECU_SEND_LOCALTIME']-df['exactdate']).astype('timedelta64[m]')<60*(j+1))]
                new = zujian[['OPT_NO','INPUT_POWER','ECU_SEND_LOCALTIME']]
                new=new.pivot('OPT_NO', 'ECU_SEND_LOCALTIME', 'INPUT_POWER')
                #new.columns
                new.reset_index()
                #if len(list(string_inverter_name))==0:
                #    continue
                #else:
                #    new['逆变器'] = list(string_inverter_name)[0]
                new['日期'] = date
                #new['关断器'] = opt
                #if len(list(substring_num)) > 0:
                #    new['子串号'] = list(substring_num)[0]
                new['组件'] = channel
                # datamppt1=pd.concat([datamppt1,new],axis=1,sort=False)
                datamppt1 = datamppt1.append(new)
            ohmygod=datamppt1[len(datamppt1.columns)-datamppt1.isnull().sum(axis=1)>=6]
            ohmygod=ohmygod.interpolate(axis=1,limit=3)
            #ohmygod.iloc[-1,:]
            ohmygod=ohmygod.fillna(method='bfill',limit=2)
            output = calcu(ohmygod)
            output_all=pd.concat([output_all,output])
                #mppt=ohmygod
                #ohmy=ohmygod[ohmygod.columns.difference(['关断器', '子串号', '日期', '组件', '逆变器'])]
                #ohmy=ohmy.iloc[:,:-3]
                #datamppt1.index.unique()
    # fig=plt.figure()
    # ax = fig.add_subplot()
    # mppt_test.plot(ax=ax)
    # plt.show()
#bbb=output_all.groupby('OPT_NO').count()
output_all.to_csv("E:/work/whynotaaa18.csv") 
        #np.array(output_all['关断器'].unique()).astype(str)