import streamlit as st
import requests
from komodo_model import KomodoEnvironmentModel

# Initialize the model
if 'model' not in st.session_state:
    st.session_state.model = KomodoEnvironmentModel()

# Function to fetch weather data from API
def get_weather_data():
    # Replace with actual API call
    return 30, 60, 5  # Example values for temperature, humidity, wind

# Streamlit app
def main():
    st.title("Komodo Dragon Environment Control")

    # Get current weather data
    if 'current_weather' not in st.session_state:
        st.session_state.current_weather = get_weather_data()

    current_temp, current_humidity, current_wind = st.session_state.current_weather

    # Display current weather
    st.header("Current Weather in Indonesia")
    st.write(f"Temperature: {current_temp:.1f}°C")
    st.write(f"Humidity: {current_humidity:.1f}%")
    st.write(f"Wind Speed: {current_wind:.1f} m/s")

    # Get model recommendations
    if 'recommended_weather' not in st.session_state:
        st.session_state.recommended_weather = st.session_state.model.get_next_weather(current_temp, current_humidity, current_wind)

    recommended_temp, recommended_humidity, recommended_wind = st.session_state.recommended_weather

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
        current_state = st.session_state.model.get_state(current_temp, current_humidity, current_wind)
        next_state = st.session_state.model.get_state(recommended_temp, recommended_humidity, recommended_wind)

        # Train the model
        action = st.session_state.model.choose_action(current_state)
        st.session_state.model.train(current_state, action, reward, next_state)

        # Update recommendations
        st.session_state.recommended_weather = st.session_state.model.get_next_weather(current_temp, current_humidity, current_wind)

        st.success("Model updated based on the Komodo dragon's reaction!")
        if st.button('Rerun'):
            st.experimental_rerun()

if __name__ == "__main__":
    main()