class ctl:
    def __init__(self, distance_coeff = -2, angle_coeff = 0.2, setDistance = 100, centre_x = 320):
        self.distance_coeff = distance_coeff
        self.angle_coeff = angle_coeff

        self.maxSpeed = 1.2
        self.maxangleSpeed = 0.2
        self.setDistance = setDistance # in cm
        self.setXcoord = centre_x
    
    # makes sure the pwm value does not 
    def clamp(self,pwm, max):
        if abs(pwm) > max:
            if pwm < 0:
                return -max
            else:
                return max
        else:
            return pwm


    def compute_speeds(self, distance, x_coord):
        error_distance = self.setDistance - distance
        error_angle = self.setXcoord - x_coord

        if distance >= self.setDistance:
            Speed = error_distance * self.distance_coeff
        else:
            Speed = 0
        
        if abs(error_angle) < 60:
            angle_speed = 0
        else:
            angle_speed = self.clamp(error_angle * self.angle_coeff, self.maxangleSpeed)

        left_pwm = Speed + angle_speed
        right_pwm = Speed - angle_speed

        return self.clamp(left_pwm, self.maxSpeed), self.clamp(right_pwm, self.maxSpeed)