import streamlit as st
import requests
from komodo_model import KomodoEnvironmentModel
import os
from dotenv import load_dotenv
# Initialize the model
model = KomodoEnvironmentModel()
load_dotenv()

# Function to fetch weather data from API
# Function to fetch weather data from API
def get_weather_data():
    # Fetch API key from environment variable
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    if not api_key:
        raise ValueError("API key not found. Please set it in the .env file.")

    # Construct the API URL
    url = f"https://api.openweathermap.org/data/2.5/weather?lat=-6.175247&lon=106.8270488&units=metric&appid={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        
        # Extract relevant weather information
        temperature = data["main"]["temp"]  # Current temperature
        humidity = data["main"]["humidity"]  # Humidity percentage
        wind_speed = data["wind"]["speed"]  # Wind speed in m/s

        return temperature, humidity, wind_speed
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None, None, None

# Streamlit app
def main():
    st.title("Komodo Dragon Environment Control")

    # Get current weather data
    current_temp, current_humidity, current_wind = get_weather_data()

    # Display current weather
    st.header("Current Weather in Indonesia")
    st.write(f"Temperature: {current_temp:.1f}°C")
    st.write(f"Humidity: {current_humidity:.1f}%")
    st.write(f"Wind Speed: {current_wind:.1f} m/s")

    # Get model recommendations
    recommended_temp, recommended_humidity, recommended_wind = model.get_next_weather(current_temp, current_humidity, current_wind)

    # Display recommendations
    st.header("Recommended Environment Settings")
    st.write(f"Temperature: {recommended_temp:.1f}°C")
    st.write(f"Humidity: {recommended_humidity:.1f}%")
    st.write(f"Wind Speed: {recommended_wind:.1f} m/s")

    # Input for Komodo dragon reaction
    reaction = st.slider("Komodo Dragon Reaction (1: Pleased, 5: Neutral, 10: Agitated)", 1, 10, 5)

    if st.button("Submit Reaction"):
        # Calculate reward (invert the scale so that 1 is the highest reward)
        reward = 11 - reaction

        # Get the states
        current_state = model.get_state(current_temp, current_humidity, current_wind)
        next_state = model.get_state(recommended_temp, recommended_humidity, recommended_wind)

        # Train the model
        action = model.choose_action(current_state)
        model.train(current_state, action, reward, next_state)

        st.success("Model updated based on the Komodo dragon's reaction!")
        st.experimental_rerun()

if __name__ == "__main__":
    main()