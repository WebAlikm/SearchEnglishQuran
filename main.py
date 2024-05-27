import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import txtai

# The rest of your code here
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

embeddings.index(verse)
embeddings.save('embeddings.tar.gz')



query = 'Mankind'
result = embeddings.search(query, limit=10)

actual_result = [verse[x[0]] for x in result]
print(actual_result)
