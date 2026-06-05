from flask import Flask, request, jsonify
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

app = Flask(__name__)

def least_square(data, future_periods):
    n = len(data)
    y = np.array(data, dtype=float)

    # X simetris: -11, -9, ..., 9, 11 (untuk n=12)
    x = np.array([2 * i - (n - 1) for i in range(n)], dtype=float)

    # Karena ΣX = 0
    a = np.sum(y) / n
    b = np.sum(x * y) / np.sum(x ** 2)

    # Fitted values
    fitted = (a + b * x).tolist()

    # Prediksi ke depan
    last_x = x[-1]
    predictions = [round(a + b * (last_x + 2 * m), 2) for m in range(1, future_periods + 1)]

    return {
        'a': round(float(a), 4),
        'b': round(float(b), 4),
        'x_values': x.tolist(),
        'fitted': [round(v, 2) for v in fitted],
        'predictions': predictions,
    }

def double_exponential_smoothing(data, future_periods, alpha=0.3):
    n = len(data)
    s1 = [0.0] * n
    s2 = [0.0] * n
    s1[0] = data[0]
    s2[0] = data[0]

    for t in range(1, n):
        s1[t] = alpha * data[t] + (1 - alpha) * s1[t - 1]
        s2[t] = alpha * s1[t]   + (1 - alpha) * s2[t - 1]

    coeff = alpha / (1 - alpha)
    at = 2 * s1[-1] - s2[-1]
    bt = coeff * (s1[-1] - s2[-1])

    fitted = []
    for t in range(n):
        if t == 0:
            fitted.append(round(data[0], 2))
        else:
            prev_at = 2 * s1[t-1] - s2[t-1]
            prev_bt = coeff * (s1[t-1] - s2[t-1])
            fitted.append(round(prev_at + prev_bt, 2))

    predictions = [round(at + bt * m, 2) for m in range(1, future_periods + 1)]

    return {
        'alpha': alpha,
        'at': round(at, 4),
        'bt': round(bt, 4),
        'fitted': fitted,
        'predictions': predictions,
    }

def calculate_accuracy(actual, forecast):
    n = len(actual)
    sum_abs = sum_sq = sum_pct = 0
    valid_n = 0

    for i in range(n):
        error = actual[i] - forecast[i]
        sum_abs += abs(error)
        sum_sq  += error ** 2
        if actual[i] != 0:
            sum_pct += abs(error / actual[i])
            valid_n += 1

    return {
        'mad':  round(sum_abs / n, 4),
        'mse':  round(sum_sq  / n, 4),
        'mape': round((sum_pct / valid_n) * 100, 4) if valid_n > 0 else 0,
    }

@app.route('/predict', methods=['POST'])
def predict():
    body          = request.get_json()
    data          = body['data']
    future_periods = body['periods']

    ls  = least_square(data, future_periods)
    des = double_exponential_smoothing(data, future_periods)

    ls['mad'],  ls['mse'],  ls['mape']  = calculate_accuracy(data, ls['fitted']).values()
    des['mad'], des['mse'], des['mape'] = calculate_accuracy(data, des['fitted']).values()

    return jsonify({'ls': ls, 'des': des})

if __name__ == '__main__':
    app.run(port=5000, debug=True)