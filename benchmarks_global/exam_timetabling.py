import cpmpy as cp
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def day_of_exam(course, slots_per_day):
    return course // slots_per_day


def construct_examtt_simple(nsemesters=9, courses_per_semester=6, slots_per_day=9, days_for_exams=14):
    """
    Constructs a simple exam timetabling problem instance.

    :param nsemesters: Number of semesters
    :param courses_per_semester: Number of courses per semester
    :param slots_per_day: Number of available time slots each day
    :param days_for_exams: Total number of days allocated for exams
    :return: A tuple containing a ProblemInstance object and a ConstraintOracle
    """

    total_courses = nsemesters * courses_per_semester
    total_slots = slots_per_day * days_for_exams

    # Explicitly define the parameters dictionary
    parameters = {
        'nsemesters': nsemesters,
        'courses_per_semester': courses_per_semester,
        'slots_per_day': slots_per_day,
        'days_for_exams': days_for_exams
    }

    # Variables named "variables" instead of "vars" to avoid conflicts
    variables = cp.intvar(1, total_slots, shape=(nsemesters, courses_per_semester), name="var")

    model = cp.Model()
    model += cp.AllDifferent(variables)

    # Constraints on courses of the same semester
    for semester_index, row in enumerate(variables):
        # Apply day_of_exam to each course in the semester
        exam_days = [course for course in row]
        model += cp.AllDifferent(exam_days)

    C_T = list(model.constraints)

    if model.solve():
        solution = variables.value()
        # Optionally, you can clear variables or handle the solution as needed
        # variables.clear()
    else:
        print("no solution")
        solution = None  # Assign a default value or handle as needed

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # Create abstract relations using the abstract vars
    lang = [
        AV[0] == AV[1],
        AV[0] != AV[1],
        AV[0] < AV[1],
        AV[0] > AV[1],
        AV[0] >= AV[1],
        AV[0] <= AV[1],
        day_of_exam(AV[0], slots_per_day) != day_of_exam(AV[1], slots_per_day),
        day_of_exam(AV[0], slots_per_day) == day_of_exam(AV[1], slots_per_day)
    ]

    instance = ProblemInstance(
        variables=variables,
        params=parameters,
        language=lang,
        name=f"exam_timetabling_semesters{nsemesters}_courses{courses_per_semester}_"
             f"timeslots{slots_per_day}_days{days_for_exams}"
    )

    oracle = ConstraintOracle(C_T)

    return instance, model
