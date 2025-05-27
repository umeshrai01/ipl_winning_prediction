import streamlit as st
import pickle
import pandas as pd

# Teams and cities
teams = ['Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Punjab Kings',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Lucknow Super Giants',
    'Gujarat Titans']

cities = ['Bangalore', 'Delhi', 'Mumbai', 'Kolkata', 'Hyderabad', 'Chennai',
       'Jaipur', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
       'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
       'Ahmedabad', 'Cuttack', 'Nagpur', 'Visakhapatnam', 'Pune',
       'Raipur', 'Ranchi', 'Abu Dhabi',  'Bengaluru', 'Dubai',
       'Sharjah', 'Navi Mumbai', 'Chandigarh', 'Lucknow', 'Guwahati',
       'Dharamsala', 'Mohali']

# Load the model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# App title
st.title('IPL Team Winning Predictor')

# Input sections
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', step=1, format="%d")


col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', step=1, format="%d")
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out', step=1, format="%d")

# Prediction
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    remaining_wickets = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Input dataframe
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [remaining_wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Model prediction
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    st.header(f"{batting_team} - {round(win * 100)}%")
    st.header(f"{bowling_team} - {round(loss * 100)}%")
