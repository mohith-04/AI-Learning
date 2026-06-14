# router.py

def route_query(
    user_input
):

    user_input = (
        user_input.lower()
    )

    # Attendance tool
    if (
        "attendance"
        in user_input
    ):

        return {
            "tool":
            "attendance"
        }

    # Marks tool
    elif (
        "marks"
        in user_input
        or
        "score"
        in user_input
    ):

        return {
            "tool":
            "marks"
        }

    # Calculator tool
    elif any(

        symbol
        in user_input

        for symbol in
        [
            "+",
            "-",
            "*",
            "/"
        ]
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