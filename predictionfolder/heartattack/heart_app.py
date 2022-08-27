from flask import *
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('D:/predictionfolder/heartattack/model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template('heartattack_pred.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict_proba(final)
    out='{0:.{1}f}'.format(prediction[0][1], 2)
    #output=format(x,'%')
    #{:.2%}".format(x)
    output='{:.1%}'.format(prediction[0][1])
    if out > str(0.8):
        return render_template('heartattack_pred.html',pred='Your Heart Attack occurence has reached Stage3.\nProbability of diabetes occuring is {}'.format(output), x="Please consult the doctor immediately")
    elif str(0.6) <= out < str(0.8):
        return render_template('heartattack_pred.html', pred='Your Heart Attack occurence has reached Stage2.\nProbability of diabetes occuring is {}'.format(output), x="Please consult the doctor immediately")
    if str(0.5) <= out < str(0.6):
        return render_template('heartattack_pred.html', pred='Your Heart Attack occurence has reached Stage1.\nProbability of diabetes occuring is {}'.format(output), x="Please consult the doctor immediately")
    else:
        return render_template('heartattack_pred.html', pred='Your are safe.\n Probability of Heart Attack occuring is {}'.format(output), x="Please consult Doctor if you have any issues")

if __name__ == '__main__':
    app.run(debug=True)
















