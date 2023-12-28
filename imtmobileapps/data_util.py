import openai
import settings as s

openai.api_key = s.OPENAI_API_KEY


def complete(prompt):
    # query - text-davinci-003
    response = openai.Completion.create(
        engine='gpt-3.5-turbo-instruct',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response['choices'][0]['text'].strip()
