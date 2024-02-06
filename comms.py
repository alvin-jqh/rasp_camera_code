import serial
import time
import msvcrt

class SerialCommunication:
    def __init__(self, port, baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = None

    def open_connection(self):
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"Serial connection opened on port {self.port}")
        except serial.SerialException as e:
            print(f"Failed to open serial connection on port {self.port}. Error: {e}")

    def close_connection(self):
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("Serial connection closed.")

    def write_data(self, data):
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.write(data.encode())
                print(f"Data written: {data}")
            except serial.SerialException as e:
                print(f"Failed to write data. Error: {e}")
        else:
            print("Serial connection not open. Cannot write data.")
        
    def write_speeds(self, left_speed:float, right_speed:float):
        message = f"<{left_speed}, {right_speed}>"
        self.write_data(message)

    def read_speeds(self):
        received_data = self.read_data_until()
        line = received_data.strip()
        measured_left = measured_right = None
        proximity_flag = False

        if line.startswith("<") and line.endswith(">"):
            # Remove "<" and ">" and split the values
            data_values = line[1:-1].split(',')

            if len(data_values) == 3:
                measured_left = float(data_values[0])
                measured_right = float(data_values[1])
                proximity_flag = bool(int(data_values[2]))
            else:
                print("Invalid data format. Expected 3 values separated by commas.")
        else:
            print("Invalid data format. Expected data to be enclosed in '<' and '>'.")

        return measured_left, measured_right, proximity_flag
    
    def read_data_until(self, delimiter='\n', max_chars=100):
        if self.serial_connection and self.serial_connection.is_open:
            try:
                data = ''
                char_count = 0

                # Read characters until the first '<'
                while char_count < max_chars:
                    char = self.serial_connection.read(1).decode()
                    char_count += 1
                    if char == '<':
                        data += char
                        break

                # Read characters until the specified delimiter
                while char_count < max_chars:
                    char = self.serial_connection.read(1).decode()
                    data += char
                    char_count += 1
                    if char == delimiter:
                        break

                if data:
                    print(f"Read data: {data}")
                    self.serial_connection.reset_input_buffer()
                return data
            except serial.SerialException as e:
                print(f"Failed to read data. Error: {e}")
                self.serial_connection.reset_input_buffer()
                return None
        else:
            print("Serial connection not open. Cannot read data.")
            self.serial_connection.reset_input_buffer()
            return None
        

# Example usage:
if __name__ == "__main__":
    # Replace 'COM1' with your actual serial port
    serial_comm = SerialCommunication(port='COM4', baudrate=9600, timeout=1)
    
    serial_comm.open_connection()
    # time.sleep(10)

    counter = 0
    
    while True:

        # # Writing data
        serial_comm.write_speeds(counter,counter)

        
        left, right, flag = serial_comm.read_speeds()
        print(f"Left: {left}, Right: {right}, flag: {flag}")
        # serial_comm.read_data()

        counter -= 10
        time.sleep(1)