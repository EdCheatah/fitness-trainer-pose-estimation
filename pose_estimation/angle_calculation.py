import math

def calculate_angle(a, b, c):
    # a, b, c are 2D points [x, y]; b is the vertex of the angle.

    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    # Dot product: ba . bc
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]

    # Vector magnitudes
    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    # Cosine of the angle
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)

    # Convert from radians to degrees
    angle = math.degrees(math.acos(cosine_angle))

    return angle
