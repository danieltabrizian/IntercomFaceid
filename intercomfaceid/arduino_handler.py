import serial
import time
import json
import os
import logging
import sys
import threading

class ArduinoHandler:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, retry_delay=5, event_logger=None,
                 ignored_codes=None):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        config = self.load_config()
        if config is not None:
            self.port = config.get('arduino_port', port)
            self.baudrate = config.get('arduino_baudrate', baudrate)
        else:
            self.port = port
            self.baudrate = baudrate

        self.retry_delay = retry_delay
        self.ser = None
        self.mqtt_client = None
        self.event_logger = event_logger
        # Codes that are periodic bus noise — not logged at all.
        self.ignored_codes = set(ignored_codes or [])
        self._lock = threading.Lock()
        self._last_activity = time.time()
        self._reconnecting = False
        self._disconnected_since = None
        self.connect()
        self._start_watchdog()

    def load_config(self, file_path='/data/options.json'):
        if not os.path.exists(file_path):
            logging.warning(f"Configuration file {file_path} not found.")
            return None
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing the configuration file {file_path}: {e}")
        return None

    def connect(self):
        """Attempt to connect to the Arduino, retrying indefinitely.
        If called after a disconnect and no connection after 10 minutes, exits
        the process so the HA supervisor restarts the add-on cleanly."""
        attempt = 0
        while True:
            # Hard restart after 10 minutes of failed reconnection attempts.
            with self._lock:
                disconnected_since = self._disconnected_since
            if disconnected_since is not None and time.time() - disconnected_since > 600:
                logging.error("Could not reconnect to Arduino after 10 minutes. Triggering add-on restart...")
                sys.exit(1)

            try:
                ser = serial.Serial(self.port, self.baudrate, timeout=1)
                time.sleep(2)
                with self._lock:
                    self.ser = ser
                    self._last_activity = time.time()
                    self._disconnected_since = None
                logging.info(f"Connected to Arduino on {self.port}")
                if self.event_logger is not None:
                    self.event_logger.log('arduino_connected', port=self.port)
                return
            except (serial.SerialException, OSError) as e:
                attempt += 1
                delay = min(self.retry_delay * attempt, 60)
                logging.warning(f"Failed to connect to Arduino (attempt {attempt}): {e}. Retrying in {delay}s...")
                time.sleep(delay)

    def _start_watchdog(self):
        t = threading.Thread(target=self._watchdog, daemon=True)
        t.start()

    def _watchdog(self):
        # If the port has been silent for 2 minutes, check it's still alive.
        while True:
            time.sleep(30)
            with self._lock:
                ser = self.ser
            if ser is None:
                continue
            try:
                if not ser.is_open:
                    raise serial.SerialException("Port is closed")
                # Prod the port to detect a silent disconnect (raises OSError errno 5/6 on Linux).
                _ = ser.in_waiting
                self._last_activity = time.time()
            except (serial.SerialException, OSError) as e:
                logging.warning(f"Watchdog detected serial issue: {e}. Reconnecting...")
                self.reconnect()

    def set_mqtt_client(self, mqtt_client):
        self.mqtt_client = mqtt_client

    def _code(self, line):
        for prefix in ("call:", "Received HEX:"):
            if line.startswith(prefix):
                return line[len(prefix):].strip()
        return None

    def read_command(self):
        with self._lock:
            ser = self.ser
        if ser is None or not ser.is_open:
            return ""
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                self._last_activity = time.time()
                # Drop short (<=4 hex digit) codes entirely — they're bus heartbeat
                # / noise (2480, 1180, 3080, 2400, ...). No log, no return.
                c = self._code(line)
                if c is not None and len(c) <= 4:
                    return ""
                if c is not None and c in self.ignored_codes:
                    return ""
                if line and self.event_logger is not None:
                    if line.lower() == 'unlock':
                        self.event_logger.log('door_unlocked')
                    elif line.startswith('Received HEX:'):
                        self.event_logger.log('hex_received', command=line)
                    else:
                        self.event_logger.log('serial_command', command=line)
                return line
        except (serial.SerialException, OSError) as e:
            logging.error(f"Error reading from serial: {e}")
            self.reconnect()
        return ""

    def unlock(self):
        with self._lock:
            ser = self.ser
        if ser is None or not ser.is_open:
            logging.warning("No serial connection, cannot send unlock command.")
            return
        try:
            ser.write(b"unlock\n")
            self._last_activity = time.time()
            logging.info("Sent unlock command to Arduino")
        except (serial.SerialException, OSError) as e:
            logging.error(f"Error writing to serial: {e}")
            self.reconnect()
            # Retry once after reconnect
            with self._lock:
                ser = self.ser
            if ser is not None and ser.is_open:
                try:
                    ser.write(b"unlock\n")
                    logging.info("Sent unlock command to Arduino after reconnect")
                except Exception as e2:
                    logging.error(f"Failed to send unlock command after reconnecting: {e2}")

    def reconnect(self):
        """Close the current port cleanly then reconnect.
        Guard ensures only one reconnect runs at a time — safe to call from
        both the watchdog thread and the main loop simultaneously."""
        with self._lock:
            if self._reconnecting:
                return  # already in progress, don't double up
            self._reconnecting = True
            ser = self.ser
            self.ser = None
            if self._disconnected_since is None:
                self._disconnected_since = time.time()
                if self.event_logger is not None:
                    self.event_logger.log('arduino_disconnected', port=self.port)

        logging.info("Attempting to reconnect to Arduino...")
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

        try:
            self.connect()
        finally:
            with self._lock:
                self._reconnecting = False

    def close(self):
        with self._lock:
            ser = self.ser
            self.ser = None
        if ser is not None:
            ser.close()
            logging.info("Closed serial connection to Arduino")