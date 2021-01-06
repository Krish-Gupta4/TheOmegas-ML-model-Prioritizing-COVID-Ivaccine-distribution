import numpy as np
from flask import Flask, request, jsonify, render_template
#import pickle
import os
import pandas as pd
from datewise import helperdate
#from dataframe_cov import helperfunc

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))
 #path =
print(os.getcwd())
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [str(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print("things are printing->", final_features[0])

    urljson = "https://api.covid19india.org/v4/data-" + final_features[0][0] + ".json"

    print("url_printing: ", urljson)


    path = os.getcwd()

    try:
        dfs = os.listdir(path + '/cache')
        print("list of df: ", dfs)
        fileName = "df_" + str(final_features[0][0]) + ".csv"
        print("searching filename: ", fileName)
        for df in dfs:
            if(df == fileName):
                print("File found")
                dff = pd.read_csv(r'cache/'+str(fileName))
                dff = dff.sort_values(by = ['priority_score'], axis = 0)
                rank = [i for i in range(1,641)]
                dff['rank'] = rank
                #dff.set_index("District name")
                print(dff)
                print("yes")
                output = dff[dff["District name"] == final_features[0][1]]
                active = output['active']
                death = output['deceased']
                priority_score = output['priority_score']
                rank = 641 - int(output['rank'])
                return render_template('index.html', prediction_text='{} has number of {} active patients, the number of registered death cases are {}. \n\n So according to our model, overall priority score is {} and it should be ranked {} among 640 district '.format(final_features[0][1], int(active), int(death), int(priority_score), rank))
    except:
        print("It may take time")

    df = helperdate(urljson, final_features[0][1], final_features[0][0])
    df = df.reset_index()
    #print(df)
    df = df.sort_values(by = ["priority_score"], axis = 0)
    rank = [i for i in range(1,641)]
    df['rank'] = rank
    #print(df)
    output = df[df['District name'] == final_features[0][1]]
    active = output['active']
    death = output['deceased']
    priority_score = output['priority_score']
    rank = 641 - int(output['rank'])
    return render_template('index.html', prediction_text='{} has number of {} active patients, the number of registered death cases are {}. \n\n So according to our model, overall priority score is {} and it should be ranked {} among 640 district '.format(final_features[0][1], int(active), int(death), int(priority_score), rank))

    #return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
