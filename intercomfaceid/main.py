import facial_recognition
import mqtt_handler
import arduino_handler
import time
import threading
import logging
import sys

def turn_bell_off(mqtt_client):
    """Turn the bell state OFF after 10 seconds."""
    time.sleep(10)
    mqtt_client.publish_bell_state("OFF")
    logging.info("Bell state turned OFF")

def main():
    # Configure logging to output to stdout
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Flags to enable/disable components
    enable_face_recognition = True
    enable_mqtt = True
    enable_arduino = True

    face_recognizer = None
    arduino = None
    mqtt_client = None

    # Initialize components
    if enable_face_recognition:
        face_recognizer = facial_recognition.FaceRecognizer()
    if enable_arduino:
        arduino = arduino_handler.ArduinoHandler()
    if enable_mqtt: 
        mqtt_client = mqtt_handler.MQTTHandler()

    # Set up component dependencies
    if enable_face_recognition:
        face_recognizer.set_arduino(arduino)
        face_recognizer.set_mqtt_client(mqtt_client)
    if enable_arduino:
        arduino.set_mqtt_client(mqtt_client)
    if enable_mqtt:
        mqtt_client.set_face_recognizer(face_recognizer)
        mqtt_client.set_arduino(arduino)

    # face_recognizer.learn_new_face()

    while True:

        if enable_mqtt:
            mqtt_client.process_messages()

        if enable_arduino:
            command = arduino.read_command()
            if command.startswith("call:OC594F") or command.startswith("Received HEX: 0C594F"):
                logging.info(f"Received call command: {command}")
                mqtt_client.publish_bell_state("ON")
                if enable_face_recognition:
                    face_recognizer.captureFace()

                # Start a non-blocking timer to turn off the bell after 10 seconds
                threading.Thread(target=turn_bell_off, args=(mqtt_client,), daemon=True).start()

            elif command == "unlock":
                logging.info("Received unlock command")
                # Add logic to unlock the door
            elif len(command) > 2:
                logging.warning(f"Received unknown command: {command}")

        time.sleep(1)  # Adjust the sleep time as needed

if __name__ == "__main__":
    main()