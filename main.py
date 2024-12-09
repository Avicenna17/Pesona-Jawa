from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load datasets
try:
    tourism_with_id = pd.read_csv('tourism_with_id.csv')
    tourism_rating = pd.read_csv('tourism_rating.csv')
except Exception as e:
    tourism_with_id = pd.DataFrame()
    tourism_rating = pd.DataFrame()
    print(f"Error loading data: {e}")

# Preprocess the data
if not tourism_with_id.empty and not tourism_rating.empty:
    try:
        average_ratings = (
            tourism_rating.groupby('Place_Id')['Place_Ratings']
            .mean()
            .reset_index()
        )
        average_ratings['Average_Rating'] = average_ratings['Place_Ratings'].round().astype(int)

        data = pd.merge(
            tourism_with_id,
            average_ratings[['Place_Id', 'Average_Rating']],
            on='Place_Id',
            how='left'
        )
        data = data.drop_duplicates(subset=['Place_Id'])

        # Add price ranges
        price_ranges = {
            'Low': (0, 100000),
            'Medium': (100001, 500000),
            'High': (500001, float('inf'))
        }
        data['Price_Range'] = pd.cut(
            data['Price'],
            bins=[0, 100000, 500000, float('inf')],
            labels=['Low', 'Medium', 'High']
        )

        # Encode data for similarity calculations
        data_encoded = pd.get_dummies(data[['Category', 'City', 'Average_Rating']], drop_first=True)
        data_encoded = pd.concat(
            [data_encoded, pd.get_dummies(data['Price_Range'], drop_first=True)],
            axis=1
        )
    except Exception as e:
        print(f"Error processing data: {e}")
        data = pd.DataFrame()
        data_encoded = pd.DataFrame()
else:
    data = pd.DataFrame()
    data_encoded = pd.DataFrame()

def get_recommendations(category, city, price, rating):
    if data.empty or data_encoded.empty:
        return []

    min_price, max_price = price_ranges[price]
    filtered_data = data[
        (data['Category'] == category) &
        (data['City'] == city) &
        (data['Price'] >= min_price) &
        (data['Price'] <= max_price) &
        (data['Average_Rating'] >= rating)
    ].copy()

    if filtered_data.empty:
        return []

    user_input_features = data_encoded.loc[
        (data['Category'] == category) &
        (data['City'] == city) &
        (data['Price'] >= min_price) &
        (data['Price'] <= max_price)
    ].mean().values.reshape(1, -1)

    if np.isnan(user_input_features).any():
        return []

    encoded_filtered = data_encoded.loc[filtered_data.index].fillna(0)
    similarities = cosine_similarity(user_input_features, encoded_filtered)

    filtered_data['Similarity'] = similarities.flatten()
    recommended_places = (
        filtered_data[['Place_Name', 'Similarity']]
        .sort_values(by='Similarity', ascending=False)
        .head(10)  # Limit recommendations to top 10
    )
    return recommended_places['Place_Name'].tolist()

# Flask routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/budaya')
def budaya():
    return render_template('budaya.html')

@app.route('/tamanhiburan')
def tamanhiburan():
    return render_template('tamanhiburan.html')

@app.route('/pusper')
def pusper():
    return render_template('pusper.html')

@app.route('/cagaralam')
def cagaralam():
    return render_template('cagaralam.html')

@app.route('/bahari')
def bahari():
    return render_template('bahari.html')

@app.route('/ibadah')
def ibadah():
    return render_template('ibadah.html')

@app.route('/trip', methods=['GET', 'POST'])
def trip():
    if data.empty:
        return render_template('error.html', message="Data not loaded correctly.")

    if request.method == 'POST':
        category = request.form.get('category')
        city = request.form.get('city')
        price = request.form.get('price')
        rating = request.form.get('rating')

        if not category or not city or not price or not rating:
            return jsonify({'error': 'All fields are required.'}), 400

        try:
            recommendations = get_recommendations(category, city, price, int(rating))
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        return jsonify(recommendations)

    return render_template('trip.html', categories=data['Category'].unique(), cities=data['City'].unique())

if __name__ == '__main__':
    app.run(debug=True)