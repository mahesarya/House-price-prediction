from flask import Flask,request,render_template
import model

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    input_features=[x for x in request.form.values()]
    #print(input_features)

    bath=input_features[0]
    balcony=input_features[1]
    bhk=input_features[2]
    total_sqft_int=input_features[3]
    price_per_sqft=input_features[4]
    area_type=input_features[5]
    availability=input_features[6]
    location=input_features[7]

    prediction=model.predict_house_price(bath=bath,balcony=balcony,bhk=bhk,total_sqft_int=total_sqft_int,price_per_sqft=price_per_sqft,area_type=area_type,availability=availability,location=location)

    return render_template('index.html',pridicted_value="Predicted House Price is {} lakh".format(prediction))



if __name__=='__main__':
    app.run(debug=True)
