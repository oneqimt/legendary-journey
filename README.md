# Project For Understanding ML Concepts 
#### Within this project we can load data, upsert to Pinecone and retrieve data.
#### Streamlit allows us to deploy to browser.
#### This project includes samples provided by the various ML sources.
#### Sources are listed below.


# Set up
#### Set up accounts with Pinecone, OpenAI, Streamlit, HuggingFace
#### Requires Python 3.9 and higher
#### Create a file named .env in project root
#### Add these vars

##### PINECONE_API_KEY=your_pinecone_apikey
##### PINECONE_ENV=your_pinecone_environment
##### PINECONE_INDEX_NAME=your_pinecone_index_name
##### OPENAI_API_KEY=your_openai_api_key

#### pip install -r requirements.txt
#### need dotenv package
#### python -m pip install python-dotenv

### Run
run your_script.py
### Streamlit
streamlit run your_script.py [-- script args]


# Machine Learning
Concepts and Sources used in this project

# Transformers
Transformers are a type of neural network architecture,
particularly in the field of natural language processing (NLP).
Unlike traditional neural networks that process input sequentially,
one element at a time, Transformers can process all input elements at once.
They do this by incorporating a mechanism called self-attention,
which allows them to learn relationships between different positions in a sequence.
Self-attention - disambiguation involves calculating a weighted sum of the input sequence,
where the weights are determined by a learned similarity function between
each pair of input elements. This allows the model to focus on different parts
of the input sequence, depending on the task at hand.

# Embeddings
"In machine learning, an embedding is a way of representing data
as points in n-dimensional space so that similar data points cluster together."
Embeddings are a way of representing data–almost any kind of data,
like text, images, videos, users, music, whatever–as points in space
where the locations of those points in n-dimensional space are semantically meaningful.
https://daleonai.com/embeddings-explained


# Langchain
LangChain is a framework for developing applications powered by language models. 
We believe that the most powerful and differentiated applications will not only 
call out to a language model via an API, but will also:
-Be data-aware: connect a language model to other sources of data
-Be agentic: allow a language model to interact with its environment.
https://python.langchain.com/en/latest/index.html
### Docs
https://python.langchain.com/en/latest/reference.html


# Pinecone Vector Database
### EXAMPLES
https://docs.pinecone.io/docs/examples
### OVERVIEW
https://docs.pinecone.io/docs/overview


## Vector search
Unlike traditional search methods that revolve around keywords, 
vector databases index and search through ML-generated representations of data, 
called vector embeddings, to find items most similar to the query.

## Vector embeddings
Vector embeddings are sets of numbers that represent objects. 
They are generated by embedding models trained to capture the semantic 
similarity of objects in a given set. 
Pinecone supports two kinds of vector embeddings: dense embeddings and sparse embeddings.
You need to have vector embeddings to use Pinecone.
### Dense Embeddings
https://www.pinecone.io/learn/vector-embeddings/
### Sparse embeddings
A sparse vector is a vector that contains mostly zeros, with only a few non-zero elements.
if you have a dataset with millions of features where only a few features 
are important for each data point, you can represent it as a sparse vector, 
which will take up much less memory than a dense vector.

## Vector database
A vector database indexes and stores vector embeddings for efficient management and fast retrieval. 
Unlike a standalone vector index, a vector database like Pinecone provides additional capabilities 
such as index management, data management, metadata storage and filtering, and horizontal scaling.

### Generative QA with OpenAI and using Pinecone
GenerativeQA.py

https://docs.pinecone.io/docs/gen-qa-openai

# Streamlit
Adding a widget is the same as declaring a variable.No need to write a backend, 
define routes, handle HTTP requests,connect a frontend, write HTML, CSS, JavaScript.

https://streamlit.io/
### Docs
https://docs.streamlit.io/
### My Example app
https://oneqimt-streamlit-example-streamlit-app-ee47c1.streamlit.app/
## Run
#### streamlit run your_script.py [-- script args]
## Create an app
https://docs.streamlit.io/library/get-started/create-an-app#lets-put-it-all-together
## Deploy instantly
#### Effortlessly share, manage and deploy your apps, directly from Streamlit.

# Hugging Face
#### Models, datasets
https://huggingface.co/datasets
#### Text Embeddings – Open Source
##### sentence-transformers
all-MiniLM-L6-v2

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

all-mpnet-base-v2

https://huggingface.co/sentence-transformers/all-mpnet-base-v2?utm_medium=email&_hsmi=251366668&utm_content=251366668&utm_source=hs_automation


