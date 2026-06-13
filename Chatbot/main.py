from openai import OpenAI
from dotenv import load_dotenv
import os
import time

# Load Environment Variables

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in .env file"
    )

# Create OpenAI Client

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# Chat History (Memory)

messages = [
    {
        "role": "system",
        "content": """
You are a helpful AI tutor.

Rules:
- Be beginner friendly
- Explain clearly
- Stay educational
- Avoid harmful content
- Use simple examples
"""
    }
]

# Chat Function

def ask_chatbot(user_input):

    # Save user message
    messages.append({
        "role": "user",
        "content": user_input
    })

    max_retries = 3

    for attempt in range(max_retries):

        try:

            # Streaming response
            stream = client.chat.completions.create(
                model="openai/gpt-oss-20b:free",

                messages=messages,

                temperature=0.7,

                stream=True
            )

            print("\nAI: ", end="")

            assistant_reply = ""

            # Stream token-by-token
            for chunk in stream:

                if (
                    chunk.choices
                    and chunk.choices[0].delta
                ):

                    content = (
                        chunk
                        .choices[0]
                        .delta.content
                    )

                    if content:

                        print(content, end="")
                        assistant_reply += content

            print("\n")

            # Empty response check
            if not assistant_reply.strip():
                raise ValueError(
                    "Empty response from model"
                )

            # Save assistant reply
            messages.append({
                "role": "assistant",
                "content": assistant_reply
            })

            # Keep recent history only
            if len(messages) > 20:
                messages[:] = (
                    [messages[0]]
                    + messages[-19:]
                )

            return

        except Exception as e:

            print(
                f"\nRetry {attempt + 1} failed:"
            )
            print(e)

            time.sleep(2)

    # Graceful fallback
    print(
        "\nAI: Sorry, I’m temporarily "
        "unable to respond. "
        "Please try again later.\n"
    )


# Chat Loop


print("AI CHATBOT STARTED")
print("Type 'exit' to quit")


while True:

    user_input = input("\nYou: ")

    # Exit condition
    if user_input.lower() == "exit":
        print("\nGoodbye!")
        break

    # Empty input validation
    if not user_input.strip():
        print(
            "Please enter a message."
        )
        continue

    ask_chatbot(user_input)