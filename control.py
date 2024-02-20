class ctl:
    def __init__(self, distance_coeff = -2, angle_coeff = -0.2, centre_x = 320):
        self.distance_coeff = distance_coeff
        self.angle_coeff = angle_coeff

        self.maxPWM = 255
        self.setDistance = 100 # in cm
        self.setXcoord = centre_x
    
    # makes sure the pwm value does not 
    def clamp(self,pwm):
        if abs(pwm) > self.maxPWM:
            if pwm < 0:
                return -self.maxPWM
            else:
                return self.maxPWM
        else:
            return pwm


    def compute_speeds(self, distance, x_coord):
        error_distance = self.setDistance - distance
        error_angle = self.setXcoord - x_coord

        pwmSignal = error_distance * self.distance_coeff
        angle_pwm = error_angle * self.angle_coeff

        left_pwm = pwmSignal + angle_pwm
        right_pwm = pwmSignal - angle_pwm

        return self.clamp(left_pwm), self.clamp(right_pwm)


    