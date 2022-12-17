import streamlit as st
from catboost import CatBoostRegressor
model = CatBoostRegressor()      # parameters not required.
model.load_model('airbnb_price')

st.title ("Price Airbnb Ia Model Predictor")
st.header("Please enter the parameters Of the site")

reviews = st.number_input('Number Of Reviews', 0,500)

Neighborhood = st.radio('NeighBornhood',
        ['NeukÃ¶lln', 'Pankow', 'Mitte', 'Friedrichshain-Kreuzberg',
       'Steglitz - Zehlendorf', 'Tempelhof - SchÃ¶neberg', 
       'Lichtenberg', 'Charlottenburg-Wilm.', 'Treptow - KÃ¶penick',
       'Marzahn - Hellersdorf', 'Reinickendorf', 'Spandau']
       )
Room_Type = st.radio('Room Type', ['Entire home/apt', 'Private room', 'Shared room'])

Guests_Included = st.number_input('Guests Included', 0,10)
Accomodates = st.number_input('Accomodates', 0, 10)
Property_Type = Room_Type = st.radio('Property Type', 
        ['Apartment', 'House', 'Loft', 'Serviced apartment', 'Townhouse',
       'Bed and breakfast', 'Guest suite', 'Bungalow', 'Other',
       'Condominium', 'Cabin', 'Hostel', 'Houseboat', 'Boat', 'Cottage',
       'Tiny house', 'Guesthouse', 'Villa', 'Hotel', 'Tipi', 'Tent',
       'Boutique hotel', 'Resort', 'Earth house', 'Camper/RV', 'Castle',
       'Train', 'Aparthotel', 'Cave', 'Barn', 'Hut',
       'Pension (South Korea)', 'Casa particular (Cuba)', 'Treehouse',
       'Vacation home']
       )
Bedrooms = st.number_input('Bedrooms', 0,10)
Overall_Rating = st.number_input('Overall Rating', 1, 100)
Bathrooms = st.number_input('Bathrooms',0, 3)
Beds = st.number_input('Beds', 1, 5)


data = [Guests_Included, Neighborhood, Bedrooms, 
        Bathrooms, Overall_Rating, Beds, 
        Room_Type, Property_Type, reviews,
        Accomodates
           ]
    
pred = model.predict(data)
st.write(f'The Best Price Will Be: {round(pred)} Eur')