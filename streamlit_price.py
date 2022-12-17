import streamlit as st
from catboost import CatBoostRegressor
from joblib import load

model = CatBoostRegressor()# parameters not required.
model.load_model('airbnb_price')
randomForest = load('Randomforest.joblib')


st.title ("Price Airbnb Ia Model Predictor")
st.header("Please enter the parameters Of the site")

reviews = st.number_input('Number Of Reviews', 0,500)

array_neigborhood_type = ['NeukÃ¶lln', 'Pankow', 'Mitte', 'Friedrichshain-Kreuzberg',
       'Steglitz - Zehlendorf', 'Tempelhof - SchÃ¶neberg', 
       'Lichtenberg', 'Charlottenburg-Wilm.', 'Treptow - KÃ¶penick',
       'Marzahn - Hellersdorf', 'Reinickendorf', 'Spandau']

Neighborhood = st.radio('NeighBornhood', array_neigborhood_type)

array_room_type = ['Entire home/apt', 'Private room', 'Shared room']

Room_Type = st.radio('Room Type', array_room_type)

Guests_Included = st.number_input('Guests Included', 1,10)

Accomodates = st.number_input('Accomodates', 1, 10)

array_properly_type = ['Apartment', 'House', 'Loft', 'Serviced apartment', 'Townhouse',
       'Bed and breakfast', 'Guest suite', 'Bungalow', 'Other',
       'Condominium', 'Cabin', 'Hostel', 'Houseboat', 'Boat', 'Cottage',
       'Tiny house', 'Guesthouse', 'Villa', 'Hotel', 'Tipi', 'Tent',
       'Boutique hotel', 'Resort', 'Earth house', 'Camper/RV', 'Castle',
       'Train', 'Aparthotel', 'Cave', 'Barn', 'Hut',
       'Pension (South Korea)', 'Casa particular (Cuba)', 'Treehouse',
       'Vacation home']

Property_Type = st.radio('Property Type', array_properly_type)

Bedrooms = st.number_input('Bedrooms', 1,10)

Overall_Rating = st.number_input('Overall Rating', 30, 100)

Bathrooms = st.number_input('Bathrooms',1, 3)

Beds = st.number_input('Beds', 1, 5)

data_cat = [Guests_Included, Neighborhood, Bedrooms, 
        Bathrooms, Overall_Rating, Beds, 
        Room_Type, Property_Type, reviews,
        Accomodates
           ]

pred_cat = model.predict(data_cat)

data_rand = [
        Bedrooms, Overall_Rating, Guests_Included, 
        array_properly_type.index(Property_Type), Accomodates, reviews,
        Bathrooms, array_neigborhood_type.index(Neighborhood), 
        array_room_type.index(Room_Type), Beds
           ]
pred_forest = randomForest.predict([data_rand])

st.write(f'The Best Price with CatBoost Will Be: {round(pred_cat)} Eur')

st.write(f'The Best Price RandomForest Will Be: {round(pred_forest[0])} Eur')