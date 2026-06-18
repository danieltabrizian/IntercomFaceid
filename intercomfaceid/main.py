from stream_manager import StreamManager
from face_recognizer import FaceRecognizer
from event_logger import EventLogger
from blur_calibration import BlurCalibration
import web_server
import mqtt_handler
import arduino_handler
import os
import time
import logging
import sys
import subprocess
import threading

GO2RTC_BIN = '/usr/local/bin/go2rtc'
GO2RTC_CONFIG = '/usr/src/app/go2rtc.yaml'


def _run_go2rtc():
    """Run go2rtc as a supervised subprocess — auto-restart if it ever exits, so
    the camera stream self-heals."""
    if not os.path.exists(GO2RTC_BIN):
        logging.warning('go2rtc binary not found; camera restreamer disabled.')
        return
    while True:
        try:
            logging.info('Starting go2rtc...')
            proc = subprocess.Popen([GO2RTC_BIN, '-config', GO2RTC_CONFIG])
            proc.wait()
            logging.error(f'go2rtc exited (code {proc.returncode}); restarting in 5s')
        except Exception as e:
            logging.error(f'go2rtc launch error: {e}; retrying in 5s')
        time.sleep(5)


# Hex codes that appear on the bus but are NOT a doorbell press (periodic noise
# / heartbeat). These are still logged as serial commands for tracking, but do
# not trigger a snapshot, a bell_ring, or face recognition.
IGNORED_CODES = {"2400"}


def _signal_code(command):
    """Extract the code portion of a 'call:XXXX' / 'Received HEX: XXXX' line."""
    if command.startswith("call:"):
        return command[len("call:"):].strip()
    if command.startswith("Received HEX:"):
        return command[len("Received HEX:"):].strip()
    return None


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Start the bundled camera restreamer (supervised, auto-restarting).
    threading.Thread(target=_run_go2rtc, daemon=True).start()

    enable_face_recognition = True
    enable_mqtt = True
    enable_arduino = True

    event_logger = EventLogger(data_dir='/data')
    blur_calibration = BlurCalibration(path='/data/blur_calibration.json')

    face_recognizer = None
    arduino = None
    mqtt_client = None

    if enable_face_recognition:
        # Read frames from the bundled go2rtc (MJPEG passthrough of the USB cam),
        # not motionEye. go2rtc runs in this same container.
        stream_manager = StreamManager("http://127.0.0.1:1984/api/stream.mjpeg?src=webcam",
                                       autostart=False)
        face_recognizer = FaceRecognizer(stream_manager, event_logger=event_logger,
                                         blur_calibration=blur_calibration)
    if enable_arduino:
        arduino = arduino_handler.ArduinoHandler(event_logger=event_logger,
                                                 ignored_codes=IGNORED_CODES)
    if enable_mqtt:
        mqtt_client = mqtt_handler.MQTTHandler()

    if enable_face_recognition:
        face_recognizer.set_arduino(arduino)
        face_recognizer.set_mqtt_client(mqtt_client)
    if enable_arduino:
        arduino.set_mqtt_client(mqtt_client)
    if enable_mqtt:
        mqtt_client.set_face_recognizer(face_recognizer)
        mqtt_client.set_arduino(arduino)

    web_server.start_in_thread(event_logger, face_recognizer, port=8099,
                               blur_calibration=blur_calibration)

    while True:
        if enable_mqtt:
            mqtt_client.process_messages()

        if enable_arduino:
            command = arduino.read_command()
            # Trigger face capture on any intercom signal:
            # - "call:XXXXXX" format from TCS bus
            # - "Received HEX: XXXXXX" format from Arduino sketch
            code = _signal_code(command)
            is_any_signal = code is not None and code not in IGNORED_CODES
            is_door_bell  = code in ("OC594F", "0C594F")
            if is_any_signal:
                logging.info(f"Received call command: {command}")
                if enable_mqtt and is_door_bell:
                    try:
                        mqtt_client.publish_bell_state()
                    except Exception as e:
                        logging.error(f"Error publishing bell state: {e}")

                if enable_face_recognition:
                    try:
                        face_recognizer.captureFace(run_recognition=is_door_bell)
                    except Exception as e:
                        logging.error(f"Error during face capture: {e}")

            elif command == "unlock":
                logging.info("Received unlock command")
            elif len(command) > 2:
                logging.warning(f"Received unknown command: {command}")

        time.sleep(1)

if __name__ == "__main__":
    main()
