
import pandas as pd
import numpy as np
import time
import turicreate as tc
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")
# #import scripts.data_layer as data_layer

#Load the data
customers = pd.read_csv('../data/recommend_1.csv')
transactions = pd.read_csv('../data/trx_data.csv')

# example 1: split product items
transactions['products'] = transactions['products'].apply(lambda x: [int(i) for i in x.split('|')])
transactions.head(2).set_index('customerId')['products'].apply(pd.Series).reset_index()

# example 2: organize a given table into a dataframe with customerId, single productId, and purchase count
pd.melt(transactions.head(2).set_index('customerId')['products'].apply(pd.Series).reset_index(), 
             id_vars=['customerId'],
             value_name='products') \
    .dropna().drop(['variable'], axis=1) \
    .groupby(['customerId', 'products']) \
    .agg({'products': 'count'}) \
    .rename(columns={'products': 'purchase_count'}) \
    .reset_index() \
    .rename(columns={'products': 'productId'})
s=time.time()

data = pd.melt(transactions.set_index('customerId')['products'].apply(pd.Series).reset_index(), 
             id_vars=['customerId'],
             value_name='products') \
    .dropna().drop(['variable'], axis=1) \
    .groupby(['customerId', 'products']) \
    .agg({'products': 'count'}) \
    .rename(columns={'products': 'purchase_count'}) \
    .reset_index() \
    .rename(columns={'products': 'productId'})
data['productId'] = data['productId'].astype(np.int64)

def create_data_dummy(data):
    data_dummy = data.copy()
    data_dummy['purchase_dummy'] = 1
    return data_dummy

data_dummy = create_data_dummy(data)
df_matrix = pd.pivot_table(data, values='purchase_count', index='customerId', columns='productId')
df_matrix.head()

df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())

df_matrix_norm.head()

# create a table for input to the modeling

d = df_matrix_norm.reset_index()
d.index.names = ['scaled_purchase_freq']
data_norm = pd.melt(d, id_vars=['customerId'], value_name='scaled_purchase_freq').dropna()

data_norm.head()

def normalize_data(data):
    df_matrix = pd.pivot_table(data, values='purchase_count', index='customerId', columns='productId')
    df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())
    d = df_matrix_norm.reset_index()
    d.index.names = ['scaled_purchase_freq']
    return pd.melt(d, id_vars=['customerId'], value_name='scaled_purchase_freq').dropna()

train, test = train_test_split(data, test_size = .2)

# Using turicreate library, we convert dataframe to SFrame - this will be useful in the modeling part

train_data = tc.SFrame(train)
test_data = tc.SFrame(test)

# We can define a function for this step as follows

def split_data(data):
    '''
    Splits dataset into training and test set.
    
    Args:
        data (pandas.DataFrame)
        
    Returns
        train_data (tc.SFrame)
        test_data (tc.SFrame)
    '''
    train, test = train_test_split(data, test_size = .2)
    train_data = tc.SFrame(train)
    test_data = tc.SFrame(test)
    return train_data, test_data

# lets try with both dummy table and scaled/normalized purchase table

train_data_dummy, test_data_dummy = split_data(data_dummy)
train_data_norm, test_data_norm = split_data(data_norm)

# variables to define field names
user_id = 'customerId'
item_id = 'productId'
target = 'purchase_count'
users_to_recommend = list(transactions[user_id])
n_rec = 10 # number of items to recommend
n_display = 30

popularity_model = tc.popularity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target)
# Get recommendations for a list of users to recommend (from customers file)
# Printed below is head / top 30 rows for first 3 customers with 10 recommendations each

popularity_recomm = popularity_model.recommend(users=users_to_recommend, k=n_rec)

def model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display):
    if name == 'popularity':
        model = tc.popularity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target)
    elif name == 'cosine':
        model = tc.item_similarity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target, 
                                                    similarity_type='cosine')
    elif name == 'pearson':
        model = tc.item_similarity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target, 
                                                    similarity_type='pearson')
        
    recom = model.recommend(users=users_to_recommend, k=n_rec)
    return model

# variables to define field names
# constant variables include:
user_id = 'customerId'
item_id = 'productId'
users_to_recommend = list(customers[user_id])
n_rec = 10 # number of items to recommend
n_display = 30 # to print the head / first few rows in a defined dataset

# these variables will change accordingly
name = 'popularity'
target = 'purchase_dummy'
pop_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'popularity'
target = 'scaled_purchase_freq'
pop_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

train.groupby(by=item_id)['purchase_count'].mean().sort_values(ascending=False).head(20)

# these variables will change accordingly
name = 'cosine'
target = 'purchase_count'
cos = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# these variables will change accordingly
name = 'cosine'
target = 'purchase_dummy'
cos_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'cosine'
target = 'scaled_purchase_freq'
cos_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# these variables will change accordingly
name = 'pearson'
target = 'purchase_count'
pear = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# these variables will change accordingly
name = 'pearson'
target = 'purchase_dummy'
pear_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'pearson'
target = 'scaled_purchase_freq'
pear_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# create initial callable variables

models_w_counts = [popularity_model, cos, pear]
models_w_dummy = [pop_dummy, cos_dummy, pear_dummy]
models_w_norm = [pop_norm, cos_norm, pear_norm]

names_w_counts = ['Popularity Model on Purchase Counts', 'Cosine Similarity on Purchase Counts', 'Pearson Similarity on Purchase Counts']
names_w_dummy = ['Popularity Model on Purchase Dummy', 'Cosine Similarity on Purchase Dummy', 'Pearson Similarity on Purchase Dummy']
names_w_norm = ['Popularity Model on Scaled Purchase Counts', 'Cosine Similarity on Scaled Purchase Counts', 'Pearson Similarity on Scaled Purchase Counts']

#eval_counts = tc.recommender.util.compare_models(test_data, models_w_counts, model_names=names_w_counts)

#eval_norm = tc.recommender.util.compare_models(test_data_norm, models_w_norm, model_names=names_w_norm)

users_to_recommend = list(customers[user_id])

final_model = tc.item_similarity_recommender.create(tc.SFrame(data_dummy), 
                                            user_id=user_id, 
                                            item_id=item_id, 
                
                                            similarity_type='cosine')

final_model.save("Model")
