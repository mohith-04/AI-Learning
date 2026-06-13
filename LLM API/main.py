from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# Load Environment Variables

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key missing!")

# Create OpenAI Client

# Change this line in your script
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# User Input

text = "This product is amazing"

# LLM API Call

try:

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",

        temperature=0,

        messages=[

            {
                "role": "system",
                "content": (
                    "You are an AI text analyzer. "
                    "Return ONLY valid JSON."
                )
            },

            {
                "role": "user",
                "content": f"""
Analyze the text.

Return ONLY this JSON format:

{{
    "sentiment": "",
    "emotion": ""
}}

Text:
"{text}"
"""
            }
        ]
    )

    # Extract model output
    output = response.choices[0].message.content

    # Convert JSON string to Python dictionary
    result = json.loads(output)

    # Print result
    print(result)

except json.JSONDecodeError:
    print("Invalid JSON returned by model.")

except Exception as e:
    print("Error:", e)