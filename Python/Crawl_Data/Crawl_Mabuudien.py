from lib2to3.pgen2 import driver
from tkinter import Button
import pandas as pd
from selenium import  webdriver
import time
from selenium.webdriver.common.by import By
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def Crawl_code(): 
    tbl = browser.find_element(By.XPATH, "/html/body/div[2]/div[2]/div[1]/div[2]/div[2]/section/div[2]/div[1]/div/div[4]").get_attribute('outerHTML')
    df  = pd.read_html(tbl)
    df=pd.DataFrame(df[0])
    df.rename(columns={'Đối tượng gán mã':'ward', 'Mã bưu chính':'ward_code'}, inplace=True)
    hcm_district=df.loc[df['ward_code']==df['ward']].index
    df['ward_code'].loc[hcm_district]=np.nan
    #format data
    df2 = pd.DataFrame(columns=['province','district','ward','ward code'])
    df2[['ward','ward code']] = df[['ward','ward code']]
    province_find = browser.find_element(By.ID,"ftwp-header-title")
    province = province_find.text.split(' ',1)[-1].lower()
    df2['province'] = province
    df2['district'].loc[df2[df2['ward code'].isna()].index] = df2['ward'].loc[df2[df2['ward code'].isna()].index].str.lower().str.replace(regex={"thành phố ":"","quận ":"","thị xã ":"","huyện ":""})
    df2['district'] = df2['district'].fillna(method='ffill')
    df2 = df2.dropna()

    #clean data
    df_total = pd.DataFrame()
    str_select = ['X.','P.','Xã','Phường']
    str_del = ['BCP.']
    for i in str_select:
        df_total = df_total.append(df2[df2['ward'].str.contains(i,regex=False)])
    for j in str_del:
        df_total = df_total.drop(df_total[df2['ward'].str.contains(j,regex=False)].index)
    df_total['ward'] = df_total['ward'].str.replace("X. ","").str.replace("P. ","").str.lower()
    df_total = df_total.sort_index().reset_index(drop=True)
    return df_total

browser=webdriver.Edge()
url='https://inxpress360.com/ma-buu-dien/'
browser.get(url)
time.sleep(3)
province_row=browser.find_elements(By.CSS_SELECTOR, 'a[href^="https://inxpress360.com/ma-buu-dien-"]')

url_list=[]
for i in province_row:
    url_list.append(i.get_attribute('href'))


df_total=pd.DataFrame()

for i in url_list:
    try:
        browser.get(i)
        time.sleep(3)
        df_total=pd.concat([df_total, Crawl_code()])
    except:
        print("Link error "+i)
        continue
browser.quit()
df_total.to_csv('ma_buu_dien1.csv')


