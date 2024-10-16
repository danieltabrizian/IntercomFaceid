import serial
import time
import json
import os
import logging
import sys

class ArduinoHandler:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, max_retries=30, retry_delay=5):
        # Configure logging to output to stdout
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        config = self.load_config()
        if config is not None:
            self.port = config.get('arduino_port', port)
            self.baudrate = config.get('arduino_baudrate', baudrate)
        else:
            self.port = port
            self.baudrate = baudrate

        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.ser = None  # Initialize serial object
        self.mqtt_client = None
        self.connect()

    def load_config(self, file_path='/data/options.json'):
        """Load the Home Assistant add-on configuration from options.json."""
        if not os.path.exists(file_path):
            logging.warning(f"Configuration file {file_path} not found.")
            return None

        try:
            with open(file_path, 'r') as file:
                config = json.load(file)
                return config
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing the configuration file {file_path}: {e}")
        return None

    def connect(self):
        """Attempt to connect to the Arduino via serial with retries."""
        retries = 0
        while retries < self.max_retries:
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
                time.sleep(2)  # Wait for the serial connection to initialize
                logging.info(f"Connected to Arduino on {self.port}")
                return
            except serial.SerialException as e:
                retries += 1
                logging.warning(f"Failed to connect to Arduino (attempt {retries}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay)

        logging.error(f"Failed to connect after {self.max_retries} attempts. Running in disconnected mode.")

    def set_mqtt_client(self, mqtt_client):
        self.mqtt_client = mqtt_client

    def read_command(self):
        if self.ser is None:
            logging.warning("No serial connection, cannot read command.")
            return ""

        try:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                return line
        except (serial.SerialException, OSError) as e:
            logging.error(f"Error reading from serial: {e}")
            self.reconnect()

        return ""

    def unlock(self):
        if self.ser is None:
            logging.warning("No serial connection, cannot send unlock command.")
            return

        try:
            self.ser.write(b"unlock\n")
            logging.info("Sent unlock command to Arduino")
        except (serial.SerialException, OSError) as e:
            logging.error(f"Error writing to serial: {e}")
            self.reconnect()
            try:
                self.ser.write(b"unlock\n")
            except Exception as e:
                logging.error(f"Failed to send unlock command after reconnecting: {e}")

    def reconnect(self):
        """Reconnect to the Arduino in case of failure."""
        logging.info("Attempting to reconnect to Arduino...")
        self.ser = None  # Reset the serial connection
        self.connect()

    def close(self):
        if self.ser is not None:
            self.ser.close()
            logging.info("Closed serial connection to Arduino")