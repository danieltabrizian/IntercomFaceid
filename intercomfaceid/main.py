from stream_manager import StreamManager
from face_recognizer import FaceRecognizer
from event_logger import EventLogger
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

    face_recognizer = None
    arduino = None
    mqtt_client = None

    if enable_face_recognition:
        stream_manager = StreamManager("http://homeassistant.local:9081")
        face_recognizer = FaceRecognizer(stream_manager, event_logger=event_logger)
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

    web_server.start_in_thread(event_logger, face_recognizer, port=8099)

    while True:
        if enable_mqtt:
            mqtt_client.process_messages()

        if enable_arduino:
            command = arduino.read_command()
            if command.startswith("call:OC594F") or command.startswith("Received HEX: 0C594F"):
                logging.info(f"Received call command: {command}")
                if enable_mqtt:
                    try:
                        mqtt_client.publish_bell_state()
                    except Exception as e:
                        logging.error(f"Error publishing bell state: {e}")

                if enable_face_recognition:
                    try:
                        face_recognizer.captureFace()
                    except Exception as e:
                        logging.error(f"Error recognizing face: {e}")

            elif command == "unlock":
                logging.info("Received unlock command")
            elif len(command) > 2:
                logging.warning(f"Received unknown command: {command}")

        time.sleep(1)

if __name__ == "__main__":
    main()
