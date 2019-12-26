from flask import Flask, request, jsonify
import traceback
import pandas as pd
import numpy as np
import time, json, csv, joblib
import turicreate as tc



# Your API definition
app = Flask(__name__)
user_id = 'customerId'
item_id = 'productId'

def cleanData(training_data):
    clean_column_name = []
    columns = training_data.columns
    for i in range(len(columns)):
        clean_column_name.append(columns[i].lower())
    training_data.columns = clean_column_name
    
    # Drop the irrelevant columns  as shown above
    training_data = training_data.drop(["rownumber", "customerid", "surname"], axis = 1)

    target_col = ["exited"]
    cat_cols   = training_data.nunique()[training_data.nunique() < 6].keys().tolist()
    cat_cols   = [x for x in cat_cols if x not in target_col]
    #num_cols   = [x for x in training_data.columns if x not in cat_cols + target_col]
                  
    # One-Hot encoding our categorical attributes
    list_cat = ['geography', 'gender']
    training_data = pd.get_dummies(training_data, columns = list_cat, prefix = list_cat)
    
    return training_data

def create_output(model, users_to_recommend, n_rec, print_csv=True):
    recomendation = model.recommend(users=users_to_recommend, k=n_rec)
    df_rec = recomendation.to_dataframe()
    df_rec['recommendedProducts'] = df_rec.groupby([user_id])[item_id] \
        .transform(lambda x: '|'.join(x.astype(str)))
    df_output = df_rec[['customerId', 'recommendedProducts']].drop_duplicates() \
        .sort_values('customerId').set_index('customerId')
    if print_csv:
        df_output.to_csv('/output/outputfile.csv')
        print("An output file can be found in 'output' folder with name 'outputfile.csv'")
    return df_output

@app.route('/Rrecommend', methods=['POST'])
def recommend():
    recom = tc.load_model("Models/Rrec/Model") # Load "model.pkl"
    print ('Retail Recommender Model loaded')
    if recom:
        try:
            json_ = request.get_json(force=True)
            #data  = json.loads(json_)
            query = json_['customerId']
            print(query)
            df_output = create_output(recom, query, 5, print_csv=True) 
            return df_output

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


@app.route('/Fpredict', methods=['POST'])
def predict():
    pred = joblib.load("Models/Fchurn/model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("Models/Fchurn/model_columns.pkl") # Load "model_columns.pkl"

    if pred:
        try:
            json_ = request.get_json(force=True)
            query = pd.DataFrame(json_)
            query = cleanData(query)
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = list(pred.predict(query))
            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    recom = tc.load_model("Models/Rrec/Model") # Load "model.pkl"
    print ('Retail Recommender Model loaded')
    pred = joblib.load("Models/Fchurn/model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("Models/Fchurn/model_columns.pkl") # Load "model_columns.pkl"

    app.run(port=port, debug=True)
