import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests

# load the nlp model and tfidf vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

def create_similarity():
    """
    Read the main_data.csv file, create a count matrix and calculate cosine similarity.

    Returns:
    data : DataFrame
        DataFrame containing the movie data.
    similarity : ndarray
        2D array containing the cosine similarity scores between movies.
    """
    data = pd.read_csv('main_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data, similarity

def rcmd(m):
    """
    Recommends similar movies to the provided movie.

    Args:
    m : str
        The movie for which recommendations are to be generated.

    Returns:
    list
        A list of recommended movies similar to the input movie.
    """
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l

def convert_to_list(my_list):
    """
    Convert a string representation of a list to a list of strings.

    Args:
    my_list : str
        String representation of a list.

    Returns:
    list
        A list containing strings extracted from the input string.
    """
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    """
    Get a list of movie titles.

    Returns:
    list
        A list of movie titles.
    """
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

def main():
    st.title('Movie Recommendation System')
    page = st.sidebar.selectbox("Choose a page", ["Home", "Recommendation"])
    
    if page == "Home":
        st.header("Home")
        st.write("Welcome to the Movie Recommendation System.")
        st.write("Please navigate to the 'Recommendation' page to get movie recommendations.")
    
    elif page == "Recommendation":
        st.header("Movie Recommendation")
        movie_name = st.text_input("Enter a movie name:", "")
        data, similarity = create_similarity()

        if st.button("Recommend"):
            recommended_movies = rcmd(movie_name
                                    #   , data, similarity
                                      )
            st.write("Recommended Movies:")
            if isinstance(recommended_movies, str):
                st.write(recommended_movies)
            else:
                for movie in recommended_movies:
                    st.write(movie)
                    
if __name__ == "__main__":
    main()