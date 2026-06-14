from database import (student_database)

def calculator(expression):

    try:
        return eval(expression)

    except Exception:
        return "Invalid calculation"


def get_marks(student_id):

    student = (
        student_database.get(
            student_id
        )
    )

    if not student:
        return "Student not found"

    return student["marks"]


def get_attendance(
    student_id
):

    student = (
        student_database.get(
            student_id
        )
    )

    if not student:
        return "Student not found"

    return (
        f"{student['attendance']}%"
    )