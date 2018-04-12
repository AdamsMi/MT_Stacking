import pandas as pd
import numpy as np
from sklearn.cluster import Birch
import math
import re
#from Levenshtein import ratio
from tqdm import tqdm


def add_cluster_column(train_df, test_df, n_clusters):
    train_df['source'] = 'train'
    test_df['source'] = 'test'

    total_rows = train_df.shape[0] + test_df.shape[0]

    data = pd.concat([train_df, test_df])

    #split the data between "around NYC" and "other locations"
    data_c=data[(data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9)]
    data_e=data[~((data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9))]
    #put it in matrix form
    coords=data_c.as_matrix(columns=['latitude', "longitude"])

    brc = Birch(branching_factor=100, n_clusters=n_clusters, threshold=0.01,compute_labels=True)

    brc.fit(coords)
    clusters=brc.predict(coords)
    data_c["num_cluster_"+str(n_clusters)]=clusters
    data_e["num_cluster_"+str(n_clusters)]=-1 #assign cluster label -1 for the non NYC listings
    data=pd.concat([data_c,data_e])

    print('lost: {}'.format(total_rows - data[data['source']=='train'].shape[0] - data[data['source']=='test'].shape[0]))
    return data[data['source']=='train'], data[data['source']=='test']



def cart2rho(x, y):
    rho = np.sqrt(x**2 + y**2)
    return rho


def cart2phi(x, y):
    phi = np.arctan2(y, x)
    return phi


def rotation_x(row, alpha):
    x = row['latitude']
    y = row['longitude']
    return x*math.cos(alpha) + y*math.sin(alpha)


def rotation_y(row, alpha):
    x = row['latitude']
    y = row['longitude']
    return y*math.cos(alpha) - x*math.sin(alpha)


def add_rotation(degrees, df):
    namex = "rot" + str(degrees) + "_X"
    namey = "rot" + str(degrees) + "_Y"

    df['num_' + namex] = df.apply(lambda row: rotation_x(row, math.pi/(180/degrees)), axis=1)
    df['num_' + namey] = df.apply(lambda row: rotation_y(row, math.pi/(180/degrees)), axis=1)

    return df

def calculate_lev_ratio(addr, displ):
    return 1 if ratio(addr, displ) > 0.5 else 0

def operate_on_coordinates(tr_df, te_df):
    for df in [tr_df, te_df]:
        #polar coordinates system
        df["num_rho"] = df.apply(lambda x: cart2rho(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)
        df["num_phi"] = df.apply(lambda x: cart2phi(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)
        #rotations
        for angle in tqdm([15,30,45,60], desc='rotations'):
            df = add_rotation(angle, df)

    return tr_df, te_df


def cap_share(x):
    return sum(1 for c in x if c.isupper())/float(len(x)+1)




def perform_general_feature_engineering(tr_df, te_df, nan_accepted=True):

    for df in [tr_df, te_df]:

        df["room_sum"] = df["bedrooms"] + df["bathrooms"] + 1.0
        #print(df['room_sum'].describe())
        df["price_bed"] = df["price"] / (df["bedrooms"]+1.0)
        #print(df['room_sum'].describe())
        df["price_t1"] = df["price"] / (df["room_sum"]+1.0)
        #print(df['room_sum'].describe())
        df["fold_t1"] = df["bedrooms"] / (df["room_sum"]+1.0)
        #print(df['room_sum'].describe())
        df['bath_room'] = df["bathrooms"] / (df["bedrooms"]+1.0)
        #print(df['room_sum'].describe())
        df["room_dif"] = df["bedrooms"] - df["bathrooms"]


        # count of photos #
        df["num_photos"] = df["photos"].apply(len)

        # count of "features" #
        df["num_features"] = df["features"].apply(len)

        # count of words present in description column #
        df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))

        # convert the created column to datetime object so as to extract more features
        df["created"] = pd.to_datetime(df["created"])

        # Let us extract some features like year, month, day, hour from date columns #
        df["created_year"] = df["created"].dt.year
        df["created_month"] = df["created"].dt.month
        df["created_day"] = df["created"].dt.day
        df["created_hour"] = df["created"].dt.hour
        df["weekday"] = df["created"].dt.dayofweek
        df["yearday"] = df["created"].dt.dayofyear
        df['Zero_building_id'] = df['building_id'].apply(lambda x: 1 if x == '0' else 0)
        df['log_price'] = np.log(df['price'])

        df['num_exc'] = df['description'].apply(lambda x: len(x.split('!')))
        df['num_cap_share'] = df['description'].apply(cap_share)

        df['num_nr_of_lines'] = df['description'].apply(lambda x: x.count('<br /><br />'))

        df['num_redacted'] = 0
        df['num_redacted'].ix[df['description'].str.contains('website_redacted')] = 1

        df['num_email'] = 0
        df['num_email'].ix[df['description'].str.contains('@')] = 1


        reg = re.compile(".*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?", re.S)
        def try_and_find_nr(description):
            if reg.match(description) is None:
                return 0
            return 1


        df['num_phone_nr'] = df['description'].apply(try_and_find_nr)


        df['num_lev_rat'] = df.apply(lambda row: calculate_lev_ratio(row['street_address'], row['display_address']), axis=1)

        df['num_half_bathrooms'] = ((np.round(df.bathrooms) - df.bathrooms)!=0).astype(float)

        #df = df.fillna(-1).replace(np.inf, -1).replace(-np.inf, -1)

        #print(df.isnull().values.any())
        # train_df['desc'] = train_df['description']
        # train_df['desc'] = train_df['desc'].apply(lambda x: x.replace('<p><a website_redacted ', ''))
        # train_df['desc'] = train_df['desc'].apply(lambda x: x.replace('!<br /><br />', ''))




    return tr_df, te_df



def add_top_building_categories(tr_df, te_df):
    buildings_count = tr_df['mapped'].value_counts()

    for df in [tr_df, te_df]:
        for percentile in [5, 10, 18, 25, 50]:
            name = "top_{}_building".format(percentile)
            print('adding ' + name)
            df[name] = df['mapped'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 100 - percentile)] else 0)

    return tr_df, te_df

def add_top_street_categories(tr_df, te_df):
    streets_count = tr_df['street_address'].value_counts()

    for df in [tr_df, te_df]:
        for percentile in [1,3]:
            name = "top_{}_street".format(percentile)
            print('adding ' + name)
            df[name] = df['street_address'].apply(lambda x: 1 if x in streets_count.index.values[
    streets_count.values >= np.percentile(streets_count.values, 100 - percentile)] else 0)

    return tr_df, te_df