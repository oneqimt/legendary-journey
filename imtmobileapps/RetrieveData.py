import openai
import settings as s
import pinecone
import data_util

openai.api_key = s.OPENAI_API_KEY

embed_model = "text-embedding-ada-002"

index_name = s.PINECONE_INDEX_NAME
# initialize connection to pinecone
pinecone.init(
    api_key=s.PINECONE_API_KEY,
    environment=s.PINECONE_ENV
)


# connect to index
index = pinecone.Index(index_name)
# view index stats
print(index.describe_index_stats())

# Now we search, for this we need to create a query vector xq:
# "Which training method should I use for sentence transformers when " +
# "I only have pairs of related sentences?"

# "Transformers incorporate a mechanism called self-attention, " +
#         "which allows them to learn relationships between different positions in a sequence. " +
#         "Explain Self-attention and disambiguation"
# How are embeddings utilized in large language models?

# PyTorch data loader
limit = 3750
query = (

    "What are AI agents and how does it differ from ChatGPT?"


)


def retrieve(query):
    global prompt
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
            "Answer the question based on the context below.\n\n" +
            "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(contexts[:i - 1]) +
                    prompt_end
            )
            break
        elif i == len(contexts) - 1:
            prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(contexts) +
                    prompt_end
            )
    return prompt


# first we retrieve relevant items from Pinecone
query_with_contexts = retrieve(query)
print(query_with_contexts)
# then we complete the context-infused query
response = data_util.complete(query_with_contexts)
print('RESPONSE is: ', response)

