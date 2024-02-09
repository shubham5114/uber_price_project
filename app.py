import pickle
import pandas as pd
import numpy as np
import streamlit as st
from geopy.geocoders import Nominatim
import pydeck as pdk
import json
import streamlit as st
from streamlit_lottie import st_lottie

# import requests


pickle_in = open("uber.pkl", "rb")
classifier = pickle.load(pickle_in)

colm1, colm2 = st.columns([1, 1])
with colm1:

    # Define the path to the Lottie animation JSON file in your local directory
    animation_path = "car.json"

    # Read the JSON file
    with open(animation_path, "r") as f:
        animation_json = json.load(f)

    # Display the Lottie animation
    st_lottie(
        animation_json,
        # reverse=True,  # Change the direction of the animation
        height=400,
        width=400,
        speed=1,
        loop=True,
        quality="high",
        key="car",
    )

with colm2:

    # Define the path to the Lottie animation JSON file in your local directory
    animation_path = "world.json"

    # Read the JSON file
    with open(animation_path, "r") as f:
        animation_json = json.load(f)

    # Display the Lottie animation
    st_lottie(
        animation_json,
        # reverse=True,  # Change the direction of the animation
        height=400,
        width=400,
        speed=1,
        loop=True,
        quality="high",
        key="world",
    )


def harversine(long1, long2, lat1, lat2):
    long1, long2, lat1, lat2 = map(np.radians, [long1, long2, lat1, lat2])
    diff_long = long2 - long1
    diff_lat = lat2 - lat1
    km = (
        2
        * 6371
        * np.arcsin(
            np.sqrt(
                np.sin(diff_lat / 2.0) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(diff_long / 2.0) ** 2
            )
        )
    )
    return km


def predict_fare(pickup_location, dropoff_location):
    geolocator = Nominatim(user_agent="MyApp")

    # Geocode pickup location
    location_pickup = geolocator.geocode(pickup_location)
    if location_pickup is None:
        return f"Error: Pickup location '{pickup_location}' not found"

    # Geocode dropoff location
    location_dropoff = geolocator.geocode(dropoff_location)
    if location_dropoff is None:
        return f"Error: Dropoff location '{dropoff_location}' not found"

    # Get latitude and longitude of pickup and dropoff locations
    pickup_lat, pickup_lon = location_pickup.latitude, location_pickup.longitude
    dropoff_lat, dropoff_lon = location_dropoff.latitude, location_dropoff.longitude

    # Calculate prediction using Haversine distance
    distance = harversine(pickup_lon, dropoff_lon, pickup_lat, dropoff_lat)
    prediction = classifier.predict([[distance]])

    # Adjust prediction (example: converting meters to dollars)
    fare_prediction = int(prediction * 83.12) / 4
    return fare_prediction


st.write("Enter pickup Location: ")
pickup_location = st.text_input("pickup")
st.write("Enter dropoff Location: ")
dropoff_location = st.text_input("dropoff")
fare = predict_fare(pickup_location, dropoff_location)
st.write(
    f"The predicted price from '{pickup_location}' to '{dropoff_location}' is: {fare}"
)


# ----------------------------------------map--------------------------------------------------------


geolocator = Nominatim(user_agent="MyApp")

# Define pickup and dropoff locations
pickup_location = (
    geolocator.geocode(pickup_location).latitude,
    geolocator.geocode(pickup_location).longitude,
)  # Latitude and longitude of pickup location
dropoff_location = (
    geolocator.geocode(dropoff_location).latitude,
    geolocator.geocode(dropoff_location).longitude,
)

# Create DataFrame with pickup and dropoff locations
data = pd.DataFrame(
    {
        "pickup": ["Pickup"],
        "lat": [pickup_location[0]],
        "lon": [pickup_location[1]],
        "dropoff": ["Dropoff"],
        "lat2": [dropoff_location[0]],
        "lon2": [dropoff_location[1]],
    }
)

# Define layers for pickup and dropoff markers
pickup_layer = pdk.Layer(
    "ScatterplotLayer",
    data=data,
    get_position=["lon", "lat"],
    get_color=[255, 0, 0],  # Red color for pickup marker
    get_radius=250,
    pickable=True,
)
dropoff_layer = pdk.Layer(
    "ScatterplotLayer",
    data=data,
    get_position=["lon2", "lat2"],
    get_color=[0, 0, 255],  # Blue color for dropoff marker
    get_radius=250,
    pickable=True,
)

# Define view state
view_state = pdk.ViewState(
    latitude=np.mean([pickup_location[0], dropoff_location[0]]),
    longitude=np.mean([pickup_location[1], dropoff_location[1]]),
    zoom=8,
    bearing=0,
    pitch=0,
)

# Render the map
st.pydeck_chart(
    pdk.Deck(layers=[pickup_layer, dropoff_layer], initial_view_state=view_state)
)
