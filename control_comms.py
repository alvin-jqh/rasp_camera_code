from comms import SerialCommunication
from control import ctl
import time

def main():
    pwmCTL = ctl(-2,-0.2)
    line = SerialCommunication("COM4")

    line.open_connection()

    distance = 100
    x = 640

    while True:
        left, right, flag = line.read_speeds()
        print(f"Left: {left}, Right: {right}, flag: {flag}")

        pwmL, pwmR = pwmCTL.compute_speeds(distance,x)

        line.write_speeds(pwmL, pwmR)

        time.sleep(1)

if __name__ == "__main__":
    main()