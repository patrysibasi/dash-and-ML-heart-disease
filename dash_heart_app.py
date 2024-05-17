import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from joblib import load
import numpy as np
import os

model = load("model_health1.pkl")

df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

# App layout
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# App initialization
app = dash.Dash(__name__, external_stylesheets= external_stylesheets)

# Path to CSV
csv_file = 'output_data.csv'

# Save output to csv file
def save_to_csv(age, sex, chest_pain_type, resting_bp_s, cholesterol, 
               fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_angina, oldpeak, ST_slope):
    data = {'age': [age], 
            'sex': [sex], 
            'chest pain type': [chest_pain_type],
            'resting bp s': [resting_bp_s],
            'cholesterol': [cholesterol],
            'fasting blood sugar': [fasting_blood_sugar],
            'resting ecg': [resting_ecg],
            'max heart rate': [max_heart_rate],
            'exercise angina': [exercise_angina],
            'oldpeak': [oldpeak],
            'ST slope': [ST_slope] }
    df = pd.DataFrame(data)
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)

# App layout
app.layout = html.Div(children=[
    html.H1(children="Heart Disease Prediction"),

    html.Div([
        html.H3(children="Insert Patient information"),

        # Age
        html.Div([
            html.Label('Age:'),
            dcc.Slider(
                id='age', 
                min=0,
                max=100,
                step=1,
                value=50,
                marks={i: str(i) for i in range(0, 101, 10)}
            ),
            html.Div(id='age-output-container', style={'margin-top': 20})
        ], style={'width':'33%', 
                  'display':'inline-block', 
                  'background-color': 'lightgrey', 
                  'padding': '10px', 
                  'border-radius': '5px', 
                  'margin-right': '10px'}
            ),

        # Gender
        html.Div([
            html.Label('Gender'),
            dcc.RadioItems(
                id='sex', 
                options=[
                    {'label': 'Man', 'value': '1'},
                    {'label': 'Woman', 'value': '0'}
                ],
                value='1'
            ),
            html.Div(id='sex-output-container')
            ], style={'width': '33%', 
                      'background-color': 'lightgrey', 
                      'padding': '10px', 
                      'border-radius': '5px', 
                      'margin-top': '10px'}
            ),

        # Chest Pain Type
        html.Div([
            html.Label('Chest pain type'),
            dcc.RadioItems(
                id='chest pain type',
                options=[
                    {'label': 'typical angina', 'value': '1'},
                    {'label': 'atypical angina', 'value': '2'},
                    {'label': 'non-anginal pain', 'value': '3'}, 
                    {'label': 'asymptomatic', 'value': '4'}
                ],
                value='1'
            ),
            html.Div(id='chest_pain-output-container')
            ], style={'width': '33%', 
                      'background-color': 'lightgrey', 
                      'padding': '10px', 
                      'border-radius': '5px', 
                      'margin-top': '10px'}
            ),
    
        # Resting Blood Pressure
        html.Div([
            html.Label('Level of blood pressure at resting mode in mm/HG'),
            dcc.Input(id='resting bp s', type='number', placeholder='Resting bp s'
                      ),
            html.Div(id='blood-output-container')
            ], style={'width': '33%', 
                      'background-color': 'lightgrey', 
                      'padding': '10px', 
                      'border-radius': '5px', 
                      'margin-top': '10px'}
            ),

        # Cholesterol
        html.Div([
            html.Label('Serum cholesterol in mg/dl'),
            dcc.Input(id='cholesterol', type='number', placeholder='Cholesterol'
                      ),
            html.Div(id='cholesterol-output-container')
        ],  style={'width': '33%', 
                   'background-color': 'lightgrey', 
                   'padding': '10px', 
                   'border-radius': '5px', 
                   'margin-top': '10px'}
            ),

        # Fasting Blood Sugar
        html.Div([
            html.Label('Blood sugar levels on fasting > 120 mg/dl'),
            dcc.RadioItems(
                id='fasting blood sugar',
                options=[
                    {'label': 'sugar > 120 mg/dl', 'value': '1'},
                    {'label': 'sugar < 120 mg/dl ', 'value': '0'}
                ],
                value='0'
            ),
            html.Div(id='sugar-output-container')
            ], style={'width': '33%', 
                      'background-color': 'lightgrey', 
                      'padding': '10px', 
                      'border-radius': '5px', 
                      'margin-top': '10px'}
            ),

        # Resting Electrocardiogram
       html.Div([
            html.Label('Resting electrocardiogram results'),
            dcc.RadioItems(
                id='resting ecg',
                options=[
                    {'label': 'Normal', 'value': '0'},
                    {'label': 'having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)', 'value': '1'},
                    {'label': "showing probable or definite left ventricular hypertrophy by Estes' criteria", 'value': '2'}
                ],
                value='0'
            ),
            html.Div(id='electrocardiogram-output-container')
            ], style={'width': '33%', 
                      'background-color': 'lightgrey', 
                      'padding': '10px', 
                      'border-radius': '5px', 
                      'margin-top': '10px'}
            ),

        # Max heart rave
        html.Div([
            html.Label('Maximum heart rate achieved'),
            dcc.Input(id='max heart rate', type='number', placeholder='Max heart rate'
                      ),
            html.Div(id='heart-rate-output-container')
        ],  style={'width': '33%', 
                   'background-color': 'lightgrey', 
                   'padding': '10px', 
                   'border-radius': '5px', 
                   'margin-top': '10px'}
            ),

        # Excercise angina
        html.Div([
            html.Label('Exercise induced angina'),
            dcc.RadioItems(
                id='exercise angina',
                options=[
                    {'label': 'Yes', 'value': '1'},
                    {'label': 'No', 'value': '0'}
                ],
                value='0'
            ),
            html.Div(id='excercise-angina-output-container')
            ], style={'width': '33%', 
                      'background-color': 'lightgrey', 
                      'padding': '10px', 
                      'border-radius': '5px', 
                      'margin-top': '10px'}
            ),

        # Oldpeak
        html.Div([
            html.Label('ST depression induced by exercise relative to rest'),
            dcc.Input(id='oldpeak', type='number', placeholder='Oldpeak'
                      ),
            html.Div(id='oldpeak-output-container')
        ],  style={'width': '33%', 
                   'background-color': 'lightgrey', 
                   'padding': '10px', 
                   'border-radius': '5px', 
                   'margin-top': '10px'}
            ),

        # ST slope
        html.Div([
            html.Label('The slope of the peak exercise ST segment'),
            dcc.RadioItems(
                id='ST slope',
                options=[
                    {'label': 'upsloping', 'value': '1'},
                    {'label': 'flat', 'value': '2'},
                    {'label': 'downsloping', 'value': '3'}
                ],
                value='1'
            ),
            html.Div(id='ST-slope-output-container')
            ], style={'width': '33%', 
                      'background-color': 'lightgrey', 
                      'padding': '10px', 
                      'border-radius': '5px', 
                      'margin-top': '10px'}
            ),
    
        # Submit
        html.Div([
            html.Button('Submit', id='submit-button', 
                        n_clicks=0,
                        style={'margin-top': '10px',
                            'margin-bottom': '10px'}),
            html.Div(id='submit-output-container')
        ])
    ]),

    # Status/Output Text Box
    html.Div(id='output-container', className='status'),
    
    # Prediction results table
    html.Div(id='table-container')
])

# Age update
@app.callback(
    Output('age-output-container', 'children'),
    Input('age', 'value')
)

def update_age_output(age):
    return f'Selected age: {age}'

# Save to csv callback
@app.callback(
    Output('submit-output-container', 'children'),
    Input('submit-button', 'n_clicks'),
    State('age', 'value'),
    State('sex', 'value'),
    State('chest pain type', 'value'),
    State('resting bp s', 'value'),
    State('cholesterol', 'value'),
    State('fasting blood sugar', 'value'),
    State('resting ecg', 'value'),
    State('max heart rate', 'value'),
    State('exercise angina', 'value'),
    State('oldpeak', 'value'),
    State('ST slope', 'value')
)

def update_csv(n_clicks, age, sex, chest_pain_type, resting_bp_s, cholesterol, 
               fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_angina, oldpeak, ST_slope):
    if n_clicks > 0:
        save_to_csv(age, sex, chest_pain_type, resting_bp_s, cholesterol, 
               fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_angina, oldpeak, ST_slope)
        return f'Data saved: Age: {age}, Gender: {sex}, Chest pain type: {chest_pain_type}, \n Resting bp s {resting_bp_s}, Cholesterol {cholesterol}, Fasting blood sugar {fasting_blood_sugar}, \n Max heart rate {max_heart_rate}, Exercise angina {exercise_angina}, Oldpeak {oldpeak}, ST slope {ST_slope}'

# Update Prediction Output
@app.callback(
    Output('output-container', 'children'),
    Output('table-container', 'children'),
    Input('submit-button', 'n_clicks'),
    State('age', 'value'),
    State('sex', 'value'),
    State('chest pain type', 'value'),
    State('resting bp s', 'value'),
    State('cholesterol', 'value'),
    State('fasting blood sugar', 'value'),
    State('resting ecg', 'value'),
    State('max heart rate', 'value'),
    State('exercise angina', 'value'),
    State('oldpeak', 'value'),
    State('ST slope', 'value')
)


def update_output(n_clicks, age, sex, chest_pain_type, resting_bp_s, cholesterol, 
                  fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_angina, oldpeak, ST_slope):
    if n_clicks is None:
        return "Server is ready for calculation.", None
    
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'age': [age], 
        'sex': [sex], 
        'chest pain type': [chest_pain_type],
        'resting bp s': [resting_bp_s],
        'cholesterol': [cholesterol],
        'fasting blood sugar': [fasting_blood_sugar],
        'resting ecg': [resting_ecg],
        'max heart rate': [max_heart_rate],
        'exercise angina': [exercise_angina],
        'oldpeak': [oldpeak],
        'ST slope': [ST_slope]
    })
    
    # Make predictions
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    
    # Construct the output
    prediction_text = f"Predicted class: {predictions[0]}"
    probabilities_text = f"Probabilities (0, 1): {probabilities[0]}"

    output = html.Div([
        html.Div(prediction_text),
        html.Div(probabilities_text)
    ])
    
    table = html.Table([
        html.Tr([html.Th(col) for col in input_data.columns]),
        html.Tr([html.Td(input_data.iloc[0][col]) for col in input_data.columns])
    ])
    
    return output, table

# Run server
if __name__ == '__main__':
    app.run_server(debug=True)