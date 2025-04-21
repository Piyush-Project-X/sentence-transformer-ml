#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sentence_transformers import SentenceTransformer
import numpy as np

# Choose pre-trained sentence transformer model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Sample sentences
sentences = [
    "Piyush is ML engineer with expertise in Cloud Computing!"
]

# Encode sentences
embeddings = model.encode(sentences)

# Output embeddings shape
print(f'Embeddings shape: {embeddings.shape}')

# Example embedding
print("Embedding example:\n", embeddings[0])


# In[ ]:




