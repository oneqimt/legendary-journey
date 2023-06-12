import settings as s
import openai
from datasets import load_dataset
import pinecone
from tqdm.auto import tqdm
from time import sleep

# Sample project from Pinecone.
# Retrieves data and uploads to Pinecone.
# https://docs.pinecone.io/docs/gen-qa-openai

index_name = s.PINECONE_INDEX_NAME
print("INDEX NAME is ", index_name)

# initialize connection to pinecone
pinecone.init(
    api_key=s.PINECONE_API_KEY,
    environment=s.PINECONE_ENV
)

openai.api_key = s.OPENAI_API_KEY

embed_model = "text-embedding-ada-002"

res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=embed_model
)

# dict_keys(['object', 'data', 'model', 'usage'])
# Inside 'data' we will find two records, one for each of the two sentences we just embedded.
# Each vector embedding contains 1536 dimensions
print(res.keys())
# 2
print(len(res['data']))

# (1536, 1536)
print(len(res['data'][0]['embedding']), len(res['data'][1]['embedding']))

# from Hugging Face Datasets. It contains transcribed audio from several ML and tech YouTube channels.

data = load_dataset('jamescalam/youtube-transcriptions', split='train')
# Dataset({
#     features: ['title', 'published', 'url', 'video_id', 'channel_id', 'id', 'text', 'start', 'end'],
#     num_rows: 208619
# })
print(data)

print(data[0])
# {'title': 'Training and Testing an Italian BERT - Transformers From Scratch #4',
# 'published': '2021-07-06 13:00:03 UTC', 'url': 'https://youtu.be/35Pdoyi6ZoQ',
# 'video_id': '35Pdoyi6ZoQ', 'channel_id': 'UCv83tO5cePwHMt1952IVVHw', 'id': '35Pdoyi6ZoQ-t0.0',
# 'text': 'Hi, welcome to the video.', 'start': 0.0, 'end': 9.36}

new_data = []

window = 20  # number of sentences to combine
stride = 4  # number of sentences to 'stride' over, used to create overlap

for i in tqdm(range(0, len(data), stride)):
    i_end = min(len(data) - 1, i + window)
    if data[i]['title'] != data[i_end]['title']:
        # in this case we skip this entry as we have start/end of two videos
        continue
    text = ' '.join(data[i:i_end]['text'])
    # create the new merged dataset
    new_data.append({
        'start': data[i]['start'],
        'end': data[i_end]['end'],
        'title': data[i]['title'],
        'text': text,
        'id': data[i]['id'],
        'url': data[i]['url'],
        'published': data[i]['published'],
        'channel_id': data[i]['channel_id']
    })

print(new_data[0])
# {'start': 0.0, 'end': 74.12, 'title': 'Training and Testing an Italian BERT - Transformers From Scratch #4',
# 'text': "Hi, welcome to the video. So this is the fourth video in a Transformers from Scratch mini series.
# So if you haven't been following along, we've essentially covered what you can see on the screen. So we got some data.
# We built a tokenizer with it. And then we've set up our input pipeline ready to begin actually training our model,
# which is what we're going to cover in this video. So let's move over to the code.
# And we see here that we have essentially everything we've done so far. S
# So we've built our input data, our input pipeline.
# And we're now at a point where we have a data loader,
# PyTorch data loader, ready. And we can begin training a model with it.
# So there are a few things to be aware of. So I mean, first,
# let's just have a quick look at the structure of our data.", 'id': '35Pdoyi6ZoQ-t0.0',
# 'url': 'https://youtu.be/35Pdoyi6ZoQ', 'published': '2021-07-06 13:00:03 UTC',
# 'channel_id': 'UCv83tO5cePwHMt1952IVVHw'}


# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='cosine',
        metadata_config={'indexed': ['channel_id', 'published']}
    )
# connect to index
index = pinecone.Index(index_name)
# view index stats
print(index.describe_index_stats())
# {'dimension': 1536,
#  'index_fullness': 0.0,
#  'namespaces': {'': {'vector_count': 2}},
#  'total_vector_count': 2}

# We can see the index is currently empty with a total_vector_count of 0.
# We can begin populating it with OpenAI text-embedding-ada-002 built embeddings like so:
batch_size = 100  # how many embeddings we create and insert at once
for i in tqdm(range(0, len(new_data), batch_size)):
    # find end of batch
    i_end = min(len(new_data), i + batch_size)
    meta_batch = new_data[i:i_end]
    # get ids
    ids_batch = [x['id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text'] for x in meta_batch]
    # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                pass
    embeds = [record['embedding'] for record in res['data']]
    # cleanup metadata
    meta_batch = [{
        'start': x['start'],
        'end': x['end'],
        'title': x['title'],
        'text': x['text'],
        'url': x['url'],
        'published': x['published'],
        'channel_id': x['channel_id']
    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)
