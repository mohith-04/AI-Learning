from database import student_database
# TOOL 1: CALCULATOR

def calculator(expression):

    try:
        return eval(expression)

    except Exception:
        return "Invalid calculation"

# TOOL 2: MARKS LOOKUP

def get_marks(student_id):

    student = student_database.get(
        student_id
    )

    if not student:
        return None

    return student["marks"]

# TOOL 3: ATTENDANCE LOOKUP

def get_attendance(student_id):

    student = student_database.get(
        student_id
    )

    if not student:
        return None

    return student["attendance"]

# ROUTING LOGIC
# (Mock LLM Decision

def route_query(user_input):

    user_input = user_input.lower()

    # Attendance lookup
    if "attendance" in user_input:
        return {
            "tool":
            "attendance"
        }

    # Marks lookup
    elif (
        "marks" in user_input
        or "score" in user_input
    ):
        return {
            "tool":
            "marks"
        }

    # Calculator
    elif any(
        symbol in user_input
        for symbol in
        ["+", "-", "*", "/"]
    ):
        return {
            "tool":
            "calculator",
            "expression":
            user_input
        }

    return {
        "tool":
        "unknown"
    }


student_id = 101


print(
    "STUDENT ASSISTANT BOT"
)
print(
    "Type 'exit' to quit"
)

while True:

    user_input = input("\nYou: ")

    if (user_input.lower()== "exit"):
        print(
            "\nGoodbye!"
        )
        break

    # Step 1:
    # Tool routing
    tool_call = route_query(user_input)

    tool_name = tool_call["tool"]

    # VALIDATION

    valid_tools = [
        "calculator",
        "marks",
        "attendance"
    ]

    if (tool_name not in valid_tools):
        print(
            "\nBot: Sorry, "
            "I don't understand."
        )
        continue

    # TOOL EXECUTION

    if tool_name == "attendance":

        result = get_attendance(
            student_id
        )

        print(
            f"\nBot: "
            f"Your attendance "
            f"is {result}%"
        )

    elif tool_name == "marks":

        result = get_marks(
            student_id
        )

        print(
            "\nBot: "
            "Your marks:"
        )

        for (
            subject,
            marks
        ) in result.items():

            print(
                f"{subject}: "
                f"{marks}"
            )

    elif tool_name== "calculator":

        expression = (
            tool_call[
                "expression"
            ]
        )

        result = calculator(
            expression
        )

        print(
            f"\nBot: "
            f"Answer = "
            f"{result}"
        )