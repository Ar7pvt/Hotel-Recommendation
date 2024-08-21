import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('omw-1.4')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval 


data = pd.read_csv('E:\Hotel Recomendation\Hotel_Reviews.csv')
data.head()

# Replacing "United Kingdom" with "INDIA"
data.Hotel_Address = data.Hotel_Address.str.replace("United Kingdom", "INDIA")
# Now I will split the address and pick the last word in the address to identify the country
data["countries"] = data.Hotel_Address.apply(lambda x: x.split(' ')[-1])
# print(data.countries.unique())   #printing countries


data.drop(['Additional_Number_of_Scoring',
       'Review_Date','Reviewer_Nationality',
       'Negative_Review', 'Review_Total_Negative_Word_Counts',
       'Total_Number_of_Reviews', 'Positive_Review',
       'Review_Total_Positive_Word_Counts',
       'Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score',
       'days_since_review', 'lat', 'lng'],axis=1,inplace=True)

def impute(column):
    column = column[0]
    if (type(column) != list):
        return "".join(literal_eval(column))
    else:
        return column
    
data["Tags"] = data[["Tags"]].apply(impute, axis=1)
data.head()

data['countries'] = data['countries'].str.lower()
data['Tags'] = data['Tags'].str.lower()



def recommend_hotel(location, description):
    description = description.lower()
    word_tokenize(description)
    stop_words = stopwords.words('english')
    lemm = WordNetLemmatizer()
    filtered  = {word for word in description if not word in stop_words}
    filtered_set = set()
    for fs in filtered:
        filtered_set.add(lemm.lemmatize(fs))
    
    country = data[data['countries']==location.lower()]
    country = country.set_index(np.arange(country.shape[0]))
    list1 = []; list2 = []; cos = [];
    for i in range(country.shape[0]):
        temp_token = word_tokenize(country["Tags"][i])
        temp_set = [word for word in temp_token if not word in stop_words]
        temp2_set = set()
        for s in temp_set:
            temp2_set.add(lemm.lemmatize(s))
        vector = temp2_set.intersection(filtered_set)
        cos.append(len(vector))
    country['similarity']=cos
    country = country.sort_values(by='similarity', ascending=False)
    country.drop_duplicates(subset='Hotel_Name', keep='first', inplace=True)
    country.sort_values('Average_Score', ascending=False, inplace=True)
    country.reset_index(inplace=True)
    return country[["Hotel_Name", "Average_Score", "Hotel_Address"]].head()


# Taking user input for country and description
location_input = input("Enter the country: ")
description_input = input("Enter the description of your trip: ")

# Getting hotel recommendations based on user input
recommended_hotels = recommend_hotel(location_input, description_input)
# recommended_hotels =recommend_hotel('Italy', 'I am going for a business trip')

# print(recommended_hotels.head())  # This will print data on terminal

# show data on Figuer 
plt.figure(figsize=(8, 5))

# Bar plot showing the average score for each hotel
plt.barh(recommended_hotels['Hotel_Name'], recommended_hotels['Average_Score'], color='skyblue')

# Adding titles and labels
plt.xlabel('Average Score')
plt.ylabel('Hotel Name')
plt.title(f'Top 5 Recommended Hotels in {location_input.capitalize()} for {description_input.capitalize()}')

# Invert y-axis to have the highest score on top
plt.gca().invert_yaxis()

plt.tight_layout()
# Display the plot
plt.show()














































# def recommend_hotel(location, description):
#     description = description.lower()
#     tokens = word_tokenize(description)  # Tokenize the description
#     stop_words = stopwords.words('english')
#     lemm = WordNetLemmatizer()
#     filtered = {word for word in tokens if word not in stop_words}  # Filter out stopwords
#     filtered_set = set()
#     for fs in filtered:
#         filtered_set.add(lemm.lemmatize(fs))  # Lemmatize the filtered words
    
#     country = data[data['countries'] == location.lower()]
#     country = country.set_index(np.arange(country.shape[0]))
#     list1 = []
#     list2 = []
#     cos = []
    
#     for i in range(country.shape[0]):
#         temp_token = word_tokenize(country["Tags"][i])
#         temp_set = [word for word in temp_token if word not in stop_words]
#         temp2_set = set()
#         for s in temp_set:
#             temp2_set.add(lemm.lemmatize(s))
#         vector = temp2_set.intersection(filtered_set)
#         cos.append(len(vector))
    
#     country['similarity'] = cos
#     country = country.sort_values(by='similarity', ascending=False)
#     country.drop_duplicates(subset='Hotel_Name', keep='first', inplace=True)
#     country.sort_values('Average_Score', ascending=False, inplace=True)
#     country.reset_index(inplace=True)
#     return country[["Hotel_Name", "Average_Score", "Hotel_Address"]].head()