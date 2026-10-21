from flask import Flask, render_template, request, jsonify
from predict import predict_patient

app = Flask(__name__, template_folder='.')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        pred_class, pred_probs, tg_hdl, non_hdl = predict_patient(
            float(data['hdl']), 
            float(data['ldl']), 
            float(data['tg']), 
            float(data['chol']), 
            float(data['vldl']), 
            float(data['age']), 
            float(data['bmi'])
        )
        return jsonify({
            'class': int(pred_class),
            'probs': [float(pred_probs[0]*100), float(pred_probs[1]*100), float(pred_probs[2]*100)],
            'tg_hdl': float(tg_hdl),
            'non_hdl': float(non_hdl)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
