from openai import OpenAI
from dotenv import load_dotenv
import os
import json

from tools import (
    calculator,
    get_marks,
    get_attendance
)

# ENV SETUP

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# Mock logged-in student
student_id = 101

# TOOL SCHEMA

tools = [

    {
        "type": "function",

        "function": {

            "name":
            "calculator",

            "description":
            "Perform math calculations",

            "parameters": {

                "type":
                "object",

                "properties": {

                    "expression": {

                        "type":
                        "string",

                        "description":
                        "Math expression"
                    }
                },

                "required":
                ["expression"]
            }
        }
    },

    {
        "type":
        "function",

        "function": {

            "name":
            "get_marks",

            "description":
            "Get student marks",

            "parameters": {

                "type":
                "object",

                "properties": {},

                "required": []
            }
        }
    },

    {
        "type":
        "function",

        "function": {

            "name":
            "get_attendance",

            "description":
            "Get student attendance",

            "parameters": {

                "type":
                "object",

                "properties": {},

                "required": []
            }
        }
    }
]

# CHAT LOOP


print(
    "AI STUDENT ASSISTANT"
)
print(
    "Type 'exit' to quit"
)


while True:

    user_input = input(
        "\nYou: "
    )

    if (
        user_input.lower()
        == "exit"
    ):
        print(
            "\nGoodbye!"
        )
        break

    response = (
        client.chat.completions.create(

            model=
            "gpt-4.1-mini",

            messages=[

                {
                    "role":
                    "system",

                    "content":
                    """
                    You are a
                    student assistant.

                    Use tools when needed.
                    """
                },

                {
                    "role":
                    "user",

                    "content":
                    user_input
                }
            ],

            tools=tools,
            max_tokens=300
        )
    )

    message = (
        response
        .choices[0]
        .message
    )

    tool_calls = (
        message.tool_calls
    )

    
    # TOOL EXECUTION
    

    if tool_calls:

        tool_call = (
            tool_calls[0]
        )

        function_name = (
            tool_call
            .function
            .name
        )

        arguments = json.loads(

            tool_call
            .function
            .arguments
        )

        # Calculator
        if (
            function_name
            == "calculator"
        ):

            result = (
                calculator(
                    arguments[
                        "expression"
                    ]
                )
            )

        # Marks
        elif (
            function_name
            == "get_marks"
        ):

            result = (
                get_marks(
                    student_id
                )
            )

        # Attendance
        elif (
            function_name
            == "get_attendance"
        ):

            result = (
                get_attendance(
                    student_id
                )
            )

        print(
            f"\nBot: "
            f"{result}"
        )

    else:

        print(
            "\nBot:",
            message.content
        )