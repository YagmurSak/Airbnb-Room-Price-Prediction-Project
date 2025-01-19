import streamlit as st
import pandas as pd
import plotly
import plotly.express as px
import joblib

st.set_page_config(layout="wide", page_title="Panda" , page_icon="🏠")

st.title(":rainbow[Airbnb Room Price Prediction]💒")
#st.subheader(":blue[Find the best price for your home!]")
st.sidebar.title(":blue[About us]")
st.sidebar.write("""Panda was founded in 2024 with a mission to help Airbnb hosts price their properties accurately.
We recognized the challenges of inconsistent pricing and set out to provide data-driven, innovative pricing solutions.

In a short time, our advanced algorithms and market analysis have enabled us to offer reliable and effective pricing 
recommendations to hosts. Today, Panda continues to help hosts increase their earnings by providing accurate pricing
and streamlining the pricing process. Our goal is to expand our solutions and support even more hosts in the future.""")

st.sidebar.title(":blue[Contact]")
st.sidebar.write("panda@gmail.com")

home_tab, data_tab, recommendation_tab = st.tabs(["Home", "Data", "Recommendation"])

text_col, image_col_left, image_col_middle, image_col_right = home_tab.columns([2,1,1,1],gap="small")

text_col.subheader(":blue[Unlock Your Airbnb Potential!]")

text_col.markdown("###### Curious about how much you could earn by renting out your space?")

text_col.markdown("""Our smart pricing tool provides accurate estimates to 
help you set the perfect price. Maximize your income, attract more guests, and take the guesswork out of hosting.
Get started today and discover your space’s true value!

At Panda, we eliminate the guesswork in pricing your Airbnb property.
With advanced algorithms and data-driven insights, we empower hosts to unlock their rental’s full potential
by setting the ideal price.

Our platform analyzes market trends, local demand, and your property’s unique features to deliver precise,
#actionable pricing recommendations tailored to you.

Whether you’re a seasoned host or just starting out, Panda makes pricing effortless and maximizes your earnings.
#Stay ahead in the competitive short-term rental market with smarter pricing strategies. At Panda, pricing smarter
isn’t just easy it’s the key to earning more!""")

image_col_right.image("https://media.istockphoto.com/id/1255835530/tr/foto%C4%9Fraf/modern-%C3%B6zel-suburban-ev-d%C4%B1%C5%9F.jpg?s=2048x2048&w=is&k=20&c=dfbBFzdY0ghpwcgqDsr6J5BuZnBhTbhXnj7dpOlxAKo=",width=250)
image_col_right.image("https://a0.muscache.com/im/pictures/hosting/Hosting-1003185235835253133/original/40d71d60-7a1d-44a1-ae9a-476e68ca34b6.jpeg?im_w=720",width=250)
#image_col_right.image("https://a0.muscache.com/im/pictures/miso/Hosting-46441717/original/2b2100b6-3f44-47e4-9bc1-c2ed9edbc95f.jpeg?im_w=720",width=250)
image_col_left.image("https://a0.muscache.com/im/pictures/miso/Hosting-18172142/original/14317ad8-362b-43c1-99e1-ba7082554302.jpeg?im_w=720",width=250)
image_col_left.image("https://a0.muscache.com/im/pictures/miso/Hosting-1044485172677898752/original/f38cb34d-7420-4c56-8461-6f89fc84d0f9.jpeg",width=250)
image_col_middle.image("https://media.istockphoto.com/id/165493611/tr/foto%C4%9Fraf/residential-architecture-in-astoria-queens-new-york-city-family-homes.jpg?s=612x612&w=0&k=20&c=tyAr7rgpSKVKWnhZYmuPuYxuidVYkP--2Z8SXQgdrQg=",width=250)
image_col_middle.image("https://a0.muscache.com/im/pictures/miso/Hosting-900031859587250941/original/e4fb68cf-2df5-4b4e-a173-24c8d6de6054.jpeg?im_w=720&width=720&quality=70&auto=webp",width=250)
#image_col_middle.image("https://a0.muscache.com/im/pictures/1ff6d909-5ba6-42f3-9d2c-fa2327780936.jpg",width=250)


st.logo("logo.jpg",
      icon_image="logo.jpg",
    size="large")

## DATA TAB

@st.cache_data
def get_data():
    df = pd.read_csv("AB_NYC_2019.csv")
    return df

df = get_data()
df.head()
#data_tab.dataframe(df)

avg_price = df["price"].mean()
avg_price1 = int(avg_price)
formatted_price = f"${avg_price1:,}"  # Virgüllerle biçimlendirilmiş ve $ eklenmiş hali
number_of_private = len(df[df["room_type"] == "Private room"])
number_of_entire = len(df[df["room_type"] == "Entire home/apt"])
number_of_Shared = len(df[df["room_type"] == "Shared room"])

price_mean_neigborhood = df.groupby("neighbourhood")["price"].mean().reset_index()

col1, col2, col3, col4 = data_tab.columns(4)

col1.metric(":red[Average Price]", formatted_price)
col2.metric(":red[Number of Private Rooms]", number_of_private)
col3.metric(":red[Number of Shared Rooms]", number_of_Shared)
col4.metric(":red[Number of Entire home/apt]", number_of_entire) 


## GRAPH

data_tab.title("New York City Airbnb Houses Map")

neighborhood_group = data_tab.selectbox(
    "Brooklyn",
    options=df["neighbourhood_group"].unique(),
    index=0)

filtered_data = df[df["neighbourhood_group"] == neighborhood_group]
data_tab.map(filtered_data[["latitude","longitude"]])

## 2.GRAPH

price_mean_neigborhood = price_mean_neigborhood.sort_values("price", ascending=False)

fig1 = px.bar(price_mean_neigborhood,
             x="neighbourhood",
             y="price",
             title="Average Price by Neighborhood",
             labels={"Neighborhood": "Neighborhood", "price": "Average Price"},
             color="price",
             template="plotly")

data_tab.plotly_chart(fig1)


############ RECOMMENDATION TAB  ##########################
#################################################

@st.cache_data
def veri_yukle():
    return pd.read_csv("yeni_dosya1.csv")

@st.cache_resource
def model_yukle():
    return joblib.load("yeni_cikti_catboost.pkl")

df = veri_yukle()
model = model_yukle()

#TITLE
recommendation_tab.title("🏠 NYC Airbnb Room Price Prediction")
recommendation_tab.subheader("Get estimated price by area group, room type and minimum nights information!")


recommendation_tab.write("### Enter Your Details")


#######
bolge_grubu = recommendation_tab.selectbox(
    "Choose your neighbourhood group:",
    options=["Brooklyn", "Manhattan", "Queens","Staten Island", "Bronx"],
    help="Mülkün bulunduğu genel bölgeyi seçin (örn: Manhattan, Brooklyn).")

######
oda_tipi = recommendation_tab.selectbox(
    "Choose your room type:",
    options= ["Private room", "Entire home/apt", "Shared room"],
    help="Mülkün oda tipini seçin."
)

######
min_gece = recommendation_tab.number_input(
    "Enter the minimum number of nights:",
    min_value=1,
   max_value=365,
    value=1,
    step=1,
    help="Misafirlerin en az kaç gece kalması gerektiğini belirtin.")

######
puan_sayısı = recommendation_tab.number_input(
    "Enter your number of reviews:",
    min_value=1,
    max_value=365,
    value=1,
    step=1,
    help="Airbnb uygulamasında odanızın puan sayısını seçin.")

#####
oda_sayısı = recommendation_tab.number_input(
    "Enter your number of rooms:",
    min_value=1,
    max_value=365,
    value=1,
    step=1,
    help="Kiraladığınız toplam oda sayısını seçin."
)


#####
musait = recommendation_tab.number_input(
    "Enter the number of days your room is available per year:",
    min_value=1,
    max_value=365,
    value=1,
    step=1,
    help="Odanızın yılda toplam kaç gün müsait olduğunu seçin."
)


######
if recommendation_tab.button("Calculate estimated price"):
    input_data = {
        "neighbourhood_group": bolge_grubu,
        "minimum_nights": min_gece * 0.01,
        "room_type_Private room": 1 if oda_tipi == "Private room" else 0,
        "room_type_Shared room": 1 if oda_tipi == "Shared room" else 0,
        #"neighbourhood":mahalle,
        "number_of_reviews":puan_sayısı,
        "calculated_host_listings_count":oda_sayısı,
        "availability_365":musait,
        #"reviews_per_month":aylık_puan_sayısı
        }

    for group in ["Brooklyn", "Manhattan", "Queens", "Staten Island"]:
        input_data[f"neighbourhood_group_{group}"] = 1 if bolge_grubu == group else 0

    input_df = pd.DataFrame([input_data])

    feature_names = model.feature_names_


    for col in feature_names:
        if col not in input_df:
            input_df[col] = 0 

    input_df = input_df[feature_names]


    tahmini_fiyat = model.predict(input_df)[0]


    recommendation_tab.success(f"Calculated Price: {tahmini_fiyat:.2f} USD")
