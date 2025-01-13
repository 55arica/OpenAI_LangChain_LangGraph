from openai import OpenAI
client = OpenAI(api_key="")


chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. give answer in one line",
        },
        {
            "role": "user",
            "content": "Who is BTS?",
        }
    ],
    model="gpt-4-1106-preview",
    max_tokens = 300,
    temperature = 0.8,
)

chat_completion.choices[0].message.content
