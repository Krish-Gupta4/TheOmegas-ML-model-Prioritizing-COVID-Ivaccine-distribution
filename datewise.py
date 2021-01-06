import json
import pandas as pd
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

import mcdm

def helperdate(loc, dist, date):
    def getJson(urlLoc):
        with urlopen(urlLoc) as rsp:
            gotRaw = rsp.read()

        data = json.loads(gotRaw)
        return data


    location = loc
    data = getJson(location)
    data.keys()

    df_district_data = pd.DataFrame(data)
    #df_district_data = df_district_data.transpose()
    #df_district_data.head()

    df_district_data = df_district_data[df_district_data.index == "districts"]
    df_district_data = df_district_data.transpose()

    def max_active_depot(d,func):
#     del d[func]
        maxi = 0
        maxi_district = 0
        for i in list(d.keys()):
            try:
                if (maxi < d[i]['delta']['confirmed']):
                    maxi = d[i]['delta']['confirmed']
                    maxi_district = i
            except:
                continue
        return maxi_district



    df_cov = pd.DataFrame({})
    df_cov['district'] = 0
    df_cov['active'] = 0
    df_cov['confirmed'] = 0
    df_cov['deceased'] = 0
    df_cov['recovered'] = 0

    count =0
    start = 0
    for i in range(len(df_district_data.districts)):
        try:
            if ("Unknown" in list(df_district_data.districts[i].keys()) and len(df_district_data.districts[i].keys())!=1):
                    print("-----------------------------------")
                    max_district = max_active_depot(df_district_data.districts[i].copy(),"Unknown")
                    print(max_district)
                    df_district_data.districts[i][max_district]['delta']['confirmed'] = df_district_data.districts[i][max_district]['delta']['confirmed']+df_district_data.districts[i]["Unknown"]['delta']['confirmed']
                    df_district_data.districts[i][max_district]['total']['confirmed'] = df_district_data.districts[i][max_district]['total']['confirmed']+df_district_data.districts[i]["Unknown"]['confirmed']
                    df_district_data.districts[i][max_district]['total']['deceased'] = df_district_data.districts[i][max_district]['total']['deceased']+df_district_data.districts[i]["Unknown"]['deceased']
                    df_district_data.districts[i][max_district]['total']['recovered'] = df_district_data.districts[i][max_district]['total']['recovered']+df_district_data.districts[i]["Unknown"]['recovered']

            for keys,values in df_district_data.districts[i].items():
                #print("keys:->", keys)
                values_total = values ['total']
                #print("values total :->",values_total)

                try:
                    values_active = values['delta']
                except:
                    values['delta']= {'confirmed':0}
                    values_active = values['delta']

                if (type(values['delta']) == int):
                    count =count+1
                    values['delta'] = {'confirmed':0}

                if ('confirmed' not in values_total.keys()):
                    values_total['confirmed'] = 0
                if ('deceased' not in values_total.keys()):
                    values_total['deceased'] = 0
                if ('recovered' not in values_total.keys()):
                    values_total['recovered'] = 0
                if ('confirmed' not in values['delta'].keys() ):
                    values['delta']={'confirmed':0}

                df_cov.loc[start] = [keys,values_active['confirmed'],values_total['confirmed'],values_total['deceased'],values_total['recovered']]
                start = start +1
        except:
            continue

        df_census = pd.read_csv(r"datasets\india-districts-census-2011.csv")

        l = list(df_census["District name"])
        df_census['active'] = 0
        df_census['confirmed'] = 0
        df_census['deceased'] = 0
        df_census['recovered'] = 0


        l1 = list(df_census["District name"].unique())
        for i in l1:
            try:
                #print(i)
                df_census.loc[df_census["District name"] == str(i),'active']=df_cov[df_cov["district"] == str(i)].active.item()
                df_census.loc[df_census["District name"] == str(i),'confirmed'] = df_cov.loc[df_cov["district"] == str(i)].confirmed.item()
                df_census.loc[df_census["District name"] == str(i),'deceased'] = df_cov.loc[df_cov["district"] == str(i)].deceased.item()
                df_census.loc[df_census["District name"] == str(i),'recovered'] = df_cov.loc[df_cov["district"] == str(i)].recovered.item()
            except:
                continue

    df_selected = df_census

    df_selected.replace(',','', regex=True,inplace = True)

    df_selected['active'] = df_selected['active'].astype('int')
    df_selected['deceased'] = df_selected['deceased'].astype('int')
    df_selected['confirmed'] = df_selected['confirmed'].astype('int')
    df_selected['recovered'] = df_selected['recovered'].astype('int')


    df_SVI_HVI = pd.DataFrame({})

    SVI_index = pd.read_csv(r"datasets\ SVI_index.csv")
    HVI_index = pd.read_csv(r"datasets\ HVI_index.csv")

    df_SVI_HVI["District name"] = df_selected["District name"]
    df_SVI_HVI["State name"] = df_selected["State name"]
    df_SVI_HVI["SVI index"] = SVI_index["svi.eff"]
    df_SVI_HVI["HVI index"] = HVI_index["hvi.eff"]
    #pick rest of the from df_cov
    df_SVI_HVI["active"] = df_selected["active"]
    df_SVI_HVI["confirmed"] = df_selected["confirmed"]
    df_SVI_HVI["recovered"] = df_selected["recovered"]
    df_SVI_HVI["deceased"] = df_selected["deceased"]
    df_SVI_HVI['recovered'] = 1- (df_SVI_HVI['recovered']/df_SVI_HVI['recovered'].max())



    #%matplotlib inline


    standard_scaler = StandardScaler()
    df_scaled = standard_scaler.fit_transform(df_SVI_HVI.iloc[:,3:])


    #from sklearn.cluster import KMeans
    #from sklearn.datasets import make_blobs
    #from yellowbrick.cluster import KElbowVisualizer

    h_cluster = AgglomerativeClustering(n_clusters=10)
    h_cluster.fit(df_scaled)

    df = df_SVI_HVI

    l_ = np.array(h_cluster.labels_)
    df['labels'] = l_

    df.set_index('District name',inplace = True)
    df['l'] = df.index.str.lower()
    df= df.sort_values('l').drop('l', axis=1)



    def weights(df):
        x_matrix =[]
        alt_names = []
        for i in list(df.iloc[:,3:].columns.unique()):
            alt_names.append(i)
            x_matrix.append(list(df.loc[:,i]))
            d = dict(mcdm.rank(x_matrix, alt_names=alt_names, w_method="CRITIC", s_method="SAW"))
        return d

    df_0 = df[df.labels == 0].iloc[:,1:]
    df_0.drop(columns="labels",axis =1,inplace =True)
    df_1 = df[df.labels == 1].iloc[:,1:]
    df_1.drop(columns="labels",axis =1,inplace =True)
    df_2 = df[df.labels == 2].iloc[:,1:]
    df_2.drop(columns="labels",axis =1,inplace =True)
    df_3 = df[df.labels == 3].iloc[:,1:]
    df_3.drop(columns="labels",axis =1,inplace =True)
    df_4 = df[df.labels == 4].iloc[:,1:]
    df_4.drop(columns="labels",axis =1,inplace =True)
    df_5 = df[df.labels == 5].iloc[:,1:]
    df_5.drop(columns="labels",axis =1,inplace =True)
    df_6 = df[df.labels == 6].iloc[:,1:]
    df_6.drop(columns="labels",axis =1,inplace =True)
    df_7 = df[df.labels == 7].iloc[:,1:]
    df_7.drop(columns="labels",axis =1,inplace =True)
    df_8 = df[df.labels == 8].iloc[:,1:]
    df_8.drop(columns="labels",axis =1,inplace =True)
    df_9 = df[df.labels == 9].iloc[:,1:]
    df_9.drop(columns="labels",axis =1,inplace =True)
    df_10 = df[df.labels == 10].iloc[:,1:]
    df_10.drop(columns="labels",axis =1,inplace =True)


    def weight_dict(df_0):
        x_matrix =[]
        alt_names = []
        for i in list(df_0.columns.unique()):
            alt_names.append(i)
            x_matrix.append(list(df_0.loc[:,i]))
        if (len(df_0)>1):
            l = mcdm.rank(x_matrix, alt_names=alt_names, n_method="Linear2", w_method="CRITIC", s_method="SAW")
            d=dict(l)
        else:
            d ={ 'confirmed': 1,
                 'recovered': 1,
                 'HVI index': 1,
                 'SVI index': 1,
                 'active':1}
        return d

    d0 = weight_dict(df_0)
    d1 = weight_dict(df_1)
    d2 = weight_dict(df_2)
    d3 = weight_dict(df_3)
    d4 = weight_dict(df_4)
    d5 = weight_dict(df_5)
    d6 = weight_dict(df_6)
    d7 = weight_dict(df_7)
    d8 = weight_dict(df_8)
    d9 = weight_dict(df_9)
    d10 = weight_dict(df_10)

    #print("printing df_0 ", df_0)

    df_0['priority_score'] = d0['confirmed']*df_0['confirmed'] + d0['recovered']*df_0['recovered'] + d0['active']*df_0['active'] + d0['HVI index']*df_0['HVI index'] + d0['SVI index']*df_0['SVI index']
    df_1['priority_score'] = d1['confirmed']*df_1['confirmed'] + d1['recovered']*df_1['recovered'] + d1['active']*df_1['active'] + d1['HVI index']*df_1['HVI index'] + d1['SVI index']*df_1['SVI index']
    df_2['priority_score'] = d2['confirmed']*df_2['confirmed'] + d2['recovered']*df_2['recovered'] + d2['active']*df_2['active'] + d2['HVI index']*df_2['HVI index'] + d2['SVI index']*df_2['SVI index']
    df_3['priority_score'] = d3['confirmed']*df_3['confirmed'] + d3['recovered']*df_3['recovered'] + d3['active']*df_3['active'] + d3['HVI index']*df_3['HVI index'] + d3['SVI index']*df_3['SVI index']
    df_4['priority_score'] = d4['confirmed']*df_4['confirmed'] + d4['recovered']*df_4['recovered'] + d4['active']*df_4['active'] + d4['HVI index']*df_4['HVI index'] + d4['SVI index']*df_4['SVI index']
    df_5['priority_score'] = d5['confirmed']*df_5['confirmed'] + d5['recovered']*df_5['recovered'] + d5['active']*df_5['active'] + d5['HVI index']*df_5['HVI index'] + d5['SVI index']*df_5['SVI index']
    df_6['priority_score'] = d6['confirmed']*df_6['confirmed'] + d6['recovered']*df_6['recovered'] + d6['active']*df_6['active'] + d6['HVI index']*df_6['HVI index'] + d6['SVI index']*df_6['SVI index']
    df_7['priority_score'] = d7['confirmed']*df_7['confirmed'] + d7['recovered']*df_7['recovered'] + d7['active']*df_7['active'] + d7['HVI index']*df_7['HVI index'] + d7['SVI index']*df_7['SVI index']
    df_8['priority_score'] = d8['confirmed']*df_8['confirmed'] + d8['recovered']*df_8['recovered'] + d8['active']*df_8['active'] + d8['HVI index']*df_8['HVI index'] + d8['SVI index']*df_8['SVI index']
    df_9['priority_score'] = d9['confirmed']*df_9['confirmed'] + d9['recovered']*df_9['recovered'] + d9['active']*df_9['active'] + d9['HVI index']*df_9['HVI index'] + d9['SVI index']*df_9['SVI index']
    df_10['priority_score'] = d10['confirmed']*df_10['confirmed'] + d10['recovered']*df_10['recovered'] + d10['active']*df_10['active'] + d10['HVI index']*df_10['HVI index'] + d10['SVI index']*df_10['SVI index']

    pdList = [df_0,df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10]
    new_df = pd.concat(pdList)

    new_df = new_df[['priority_score', 'deceased', 'active', 'recovered']]

    file_name  =  "cache\df_" + str(date) + ".csv"
    print("saving: ", file_name)
    #print(new_df)
    new_df.to_csv(file_name)

    return new_df
