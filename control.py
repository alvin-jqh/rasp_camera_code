from simple_pid import PID
class ctl:
    def __init__(self, distance_coeffs, angle_coeffs):
        self.dKp, self.dKi, self.dKd = distance_coeffs
        self.aKp, self.aKi, self.aKd = angle_coeffs

        self.distance_ctl = PID(self.dKp, self.dKi, self.dKd, setpoint=100)
        self.angle_ctl = PID(self.aKp, self.aKi, self.aKd, setpoint=320)

    def compute_speeds(self, distance, x_coord):
        straight_line_speed = self.distance_ctl(distance)
        turning_add = self.angle_ctl(x_coord)

        left_speed = straight_line_speed + turning_add
        right_speed = straight_line_speed - turning_add

        return left_speed, right_speed
    