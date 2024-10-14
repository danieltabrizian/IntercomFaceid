import serial
import time

class ArduinoHandler:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for the serial connection to initialize
    
    def set_mqtt_client(self, mqtt_client):
        self.mqtt_client = mqtt_client

    def read_command(self):
        if self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8').strip()
            return line
        return ""

    def unlock(self):
        self.ser.write(b"unlock\n")
        print("Sent unlock command to Arduino")

    def close(self):
        self.ser.close()