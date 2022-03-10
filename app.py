from flask import Flask,request,render_template
from flask_cors import cross_origin
import pickle

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def page():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
@cross_origin()
def home():
    if request.method == 'POST':
        occ_2 = float(request.form['occ_2'])
        occ_3 = float(request.form['occ_3'])
        occ_4 = float(request.form['occ_4'])
        occ_5 = float(request.form['occ_5'])
        occ_6 = float(request.form['occ_6'])
        occ_husb_2 = float(request.form['occ_husb_2'])
        occ_husb_3 = float(request.form['occ_husb_3'])
        occ_husb_4 = float(request.form['occ_husb_4'])
        occ_husb_5 = float(request.form['occ_husb_5'])
        occ_husb_6 = float(request.form['occ_husb_6'])
        rate_marriage = float(request.form['rate_marriage'])
        age = float(request.form['age'])
        yrs_married = float(request.form['yrs_married'])
        children = float(request.form['children'])
        religious = float(request.form['religious'])
        educ = float(request.form['educ'])

        #load modle
        file_name = 'woman affair.pkl'
        scale_file = 'scaler_woman_affair.pkl'
        load_scale = pickle.load(open(scale_file,'rb'))
        load_model = pickle.load(open(file_name,'rb'))
        scale = load_scale.transform([[occ_2,occ_3,occ_4,occ_5,occ_6,occ_husb_2,occ_husb_3,occ_husb_4,occ_husb_5,occ_husb_6,
                                       rate_marriage,age,yrs_married,children,religious,educ]])
        pred = load_model.predict(scale)

        return render_template('results.html',pred=pred[0])

if __name__ == '__main__':
    app.run(debug=True)