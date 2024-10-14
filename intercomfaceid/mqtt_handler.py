import os
import paho.mqtt.client as mqtt
import json

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

        # Call the method to broadcast device types
        self.broadcast_device_types()

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
            self.arduino.unlock()
        elif msg.topic == self.recognize_face_command_topic:
            print("Received command to recognize face")
            self.face_recognizer.captureFace()

    def publish_bell_state(self, state):
        self.mqtt_client.publish(self.bell_state_topic, state)
        print(f"Published bell state: {state}")

    def broadcast_device_types(self):
        # Publish button for learning new face
        learn_new_face_payload = {
            "name": "Learn New Face",
            "command_topic": self.learn_face_command_topic,
            "device_class": "button"
        }
        self.mqtt_client.publish("homeassistant/button/learn_new_face/config", json.dumps(learn_new_face_payload), retain=True)

        # Publish button for unlocking the door
        unlock_door_payload = {
            "name": "Unlock Door",
            "command_topic": self.unlock_door_command_topic,
            "device_class": "button"
        }
        self.mqtt_client.publish("homeassistant/button/unlock_door/config", json.dumps(unlock_door_payload), retain=True)

        # Publish button for recognizing face
        recognize_face_payload = {
            "name": "Recognize Face",
            "command_topic": self.recognize_face_command_topic,
            "device_class": "button"
        }
        self.mqtt_client.publish("homeassistant/button/recognize_face/config", json.dumps(recognize_face_payload), retain=True)

        # Publish binary sensor for bell run state
        bell_state_payload = {
            "name": "Bell Run",
            "state_topic": self.bell_state_topic,
            "device_class": "motion",
            "payload_on": "ON",
            "payload_off": "OFF"
        }
        self.mqtt_client.publish("homeassistant/binary_sensor/bell_run/config", json.dumps(bell_state_payload), retain=True)

    def process_messages(self):
        # This method can be called periodically to process MQTT messages
        pass