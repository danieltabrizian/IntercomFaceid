import os
import paho.mqtt.client as mqtt

class MQTTHandler:
    def __init__(self):
        self.mqtt_broker = os.getenv("MQTT_BROKER", "core-mosquitto")
        self.mqtt_port = 1883
        self.mqtt_username = os.getenv("MQTT_USERNAME", "mqtt")
        self.mqtt_password = os.getenv("MQTT_PASSWORD", "mqtt")

        self.bell_state_topic = "homeassistant/binary_sensor/bell_run/state"
        self.learn_face_command_topic = "homeassistant/button/learn_new_face/set"
        self.unlock_door_command_topic = "homeassistant/button/unlock_door/set"
        self.recognize_face_command_topic = "homeassistant/button/recognize_face/set"

        self.mqtt_client = mqtt.Client()
        if self.mqtt_username and self.mqtt_password:
            self.mqtt_client.username_pw_set(self.mqtt_username, self.mqtt_password)

        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.mqtt_client.subscribe(self.learn_face_command_topic)
        self.mqtt_client.subscribe(self.unlock_door_command_topic)
        self.mqtt_client.subscribe(self.recognize_face_command_topic)
        self.mqtt_client.loop_start()

    def set_face_recognizer(self, face_recognizer):
        self.face_recognizer = face_recognizer
    
    def set_arduino(self, arduino):
        self.arduino = arduino

    def on_message(self, client, userdata, msg):
        if msg.topic == self.learn_face_command_topic:
            print("Received command to learn new face")
            self.face_recognizer.learn_new_face()
        elif msg.topic == self.unlock_door_command_topic:
            print("Received command to unlock door")
            # Call function to unlock door
            self.arduino.unlock()
        elif msg.topic == self.recognize_face_command_topic:
            print("Received command to recognize face")
            # Call function to recognize face
            self.face_recognizer.captureFace()

    def publish_bell_state(self, state):
        self.mqtt_client.publish(self.bell_state_topic, state)
        print(f"Published bell state: {state}")

    def process_messages(self):
        # This method can be called periodically to process MQTT messages
        # The actual message processing happens in the background due to loop_start()
        pass