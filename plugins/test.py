#!/usr/bin/python3

import openai
from datetime import date

# Define your deployment name, endpoint, and other settings
deployment_name = 'gpt-35-turbo'  # Adjust if you have a different model name
api_key = 'ffb3f4e4ea574d51b1eb34ff0ecedf3d'
api_version = '2023-07-01-preview'  # Adjust if you have a different API version
base_url = 'https://trapi.research.microsoft.com/gcr/shared'

# Get the current date
month_day = date.today().strftime("%B %d")

# Initialize the OpenAI client with Azure configuration
openai.api_key = api_key
openai.api_base = base_url
openai.api_type = 'azure'
openai.api_version = api_version

# Create a chat completion request
response = openai.ChatCompletion.create(
    engine=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "Today is " + month_day + ". Select one historical fact for the day at random and write a 2-stanza poem about it."
        }
    ],
    max_tokens=500
)

# Print the result
print(response['choices'][0]['message']['content'])
