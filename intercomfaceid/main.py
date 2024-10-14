import facial_recognition
import mqtt_handler
import arduino_handler
import time



def main():
    # Flags to enable/disable components
    enable_face_recognition = True
    enable_mqtt = True
    enable_arduino = False

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
            if command.startswith("call:"):
                print(f"Received call command: {command}")
                if enable_face_recognition:
                    face_recognizer.captureFace()
                # Add logic to handle the call command
            elif command == "unlock":
                print("Received unlock command")
                # Add logic to unlock the door

        time.sleep(1)  # Adjust the sleep time as needed

if __name__ == "__main__":
    main()