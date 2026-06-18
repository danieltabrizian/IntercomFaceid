from stream_manager import StreamManager
from face_recognizer import FaceRecognizer
from event_logger import EventLogger
from blur_calibration import BlurCalibration
import web_server
import mqtt_handler
import arduino_handler
import time
import logging
import sys

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    enable_face_recognition = True
    enable_mqtt = True
    enable_arduino = True

    event_logger = EventLogger(data_dir='/data')
    blur_calibration = BlurCalibration(path='/data/blur_calibration.json')

    face_recognizer = None
    arduino = None
    mqtt_client = None

    if enable_face_recognition:
        stream_manager = StreamManager("http://homeassistant.local:9081")
        face_recognizer = FaceRecognizer(stream_manager, event_logger=event_logger,
                                         blur_calibration=blur_calibration)
    if enable_arduino:
        arduino = arduino_handler.ArduinoHandler(event_logger=event_logger)
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
            is_any_signal = command.startswith("call:") or command.startswith("Received HEX:")
            is_door_bell  = command.startswith("call:OC594F") or command.startswith("Received HEX: 0C594F")
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
