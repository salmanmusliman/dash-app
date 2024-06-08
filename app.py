
from flask import Flask
from flask_ngrok import run_with_ngrok
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc  # Importing dash-bootstrap-components
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Upload the dataset file
from google.colab import files
uploaded = files.upload()

# Load your dataset (use the uploaded file name)
file_name = list(uploaded.keys())[0]
data = pd.read_csv(file_name)

# Clean and preprocess the data
data = data.drop(columns=['Unnamed: 5'])

# Encode country names
le = LabelEncoder()
data['GEO_NAME_SHORT_ENCODED'] = le.fit_transform(data['GEO_NAME_SHORT'])

# Ensure RATE_PER_100000_N is numeric
data = data[pd.to_numeric(data['RATE_PER_100000_N'], errors='coerce').notnull()]
data['RATE_PER_100000_N'] = pd.to_numeric(data['RATE_PER_100000_N'])

# Define features and target variable
X = data[['DIM_TIME', 'GEO_NAME_SHORT_ENCODED', 'RATE_PER_100000_NL', 'RATE_PER_100000_NU']]
y = data['RATE_PER_100000_N']

# Train a Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Calculate trends for future predictions
country_trends = data.groupby('GEO_NAME_SHORT_ENCODED').apply(
    lambda x: np.polyfit(x['DIM_TIME'], x['RATE_PER_100000_N'], 1)
).to_dict()

# Predict for future years 2021-2025
future_years = list(range(2021, 2026))
predictions = []

for year in future_years:
    for country_encoded in data['GEO_NAME_SHORT_ENCODED'].unique():
        trend = country_trends[country_encoded]
        future_data = pd.DataFrame({
            'DIM_TIME': [year],
            'GEO_NAME_SHORT_ENCODED': [country_encoded],
            'RATE_PER_100000_NL': [trend[0] * year + trend[1]],  # Use trend line for new data
            'RATE_PER_100000_NU': [trend[0] * year + trend[1]]   # Use trend line for new data
        })
        future_pred = model.predict(future_data)
        predictions.append((year, country_encoded, future_pred[0]))

# Create a DataFrame for the predictions
pred_df = pd.DataFrame(predictions, columns=['Year', 'Country_Encoded', 'Predicted'])

# Add the original actual data
actual_df = data[['DIM_TIME', 'GEO_NAME_SHORT_ENCODED', 'RATE_PER_100000_N']]
actual_df.columns = ['Year', 'Country_Encoded', 'Actual']

# Combine actual and predicted data
combined_df = pd.merge(actual_df, pred_df, how='outer', on=['Year', 'Country_Encoded'])

# Map encoded countries back to original names
combined_df['Country'] = le.inverse_transform(combined_df['Country_Encoded'])

# Create the Dash app
server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("Mortality Rate Over Time"),
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': country, 'value': country} for country in combined_df['Country'].unique()],
        value=combined_df['Country'].unique()[0],
        clearable=False
    ),
    dcc.Graph(id='mortality-rate-graph')
])

@app.callback(
    Output('mortality-rate-graph', 'figure'),
    [Input('country-dropdown', 'value')]
)
def update_graph(selected_country):
    country_data = combined_df[combined_df['Country'] == selected_country]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=country_data['Year'],
        y=country_data['Actual'],
        mode='lines+markers',
        name=f'Actual - {selected_country}'
    ))

    fig.add_trace(go.Scatter(
        x=country_data['Year'],
        y=country_data['Predicted'],
        mode='lines+markers',
        name=f'Predicted - {selected_country}',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title=f'Mortality Rate for {selected_country}',
        xaxis_title='Year',
        yaxis_title='Mortality Rate per 100,000',
        template='plotly_white'
    )

    return fig

if __name__ == '__main__':
    app.run_server()
