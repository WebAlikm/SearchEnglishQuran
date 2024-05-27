import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import numpy as np
import pandas as pd
import txtai

@st.cache_data
def load_data_and_embeddings():
    np.random.seed(1)
    df = pd.read_csv('Quran_English.csv')

    if 'Verse' in df.columns:
        df_clean = df.dropna()
        num_rows = df_clean.shape[0]
        sample_size = min(10000, num_rows)
        verse = df_clean.sample(sample_size).Verse.values
    else:
        raise ValueError("The column 'Verse' does not exist in the DataFrame")

    embeddings = txtai.Embeddings({
        'path': 'sentence-transformers/all-MiniLM-L6-v2'
    })

    embeddings.load('embeddings.tar.gz')

    return verse, embeddings

verse, embeddings = load_data_and_embeddings()

st.title('Verse of Surah Search')

query = st.text_input('Enter the Query:', '')

if st.button('Search'):
    if query:
        result = embeddings.search(query, limit=10)
        actual_result = [verse[x[0]] for x in result]

        for res in actual_result:
            st.write(res)
    else:
        st.write('Please Enter a Query')
