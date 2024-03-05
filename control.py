class ctl:
    def __init__(self, distance_coeff = -2, angle_coeff = 0.2, setDistance = 100, centre_x = 320):
        self.distance_coeff = distance_coeff
        self.angle_coeff = angle_coeff

        self.maxPWM = 255
        self.maxanglePWM = 35
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
            pwmSignal = error_distance * self.distance_coeff
        else:
            pwmSignal = 0
        
        if abs(error_angle) < 60:
            angle_pwm = 0
        else:
            angle_pwm = self.clamp(error_angle * self.angle_coeff, self.maxanglePWM)

        left_pwm = pwmSignal + angle_pwm
        right_pwm = pwmSignal - angle_pwm

        return self.clamp(left_pwm, self.maxPWM), self.clamp(right_pwm, self.maxPWM)