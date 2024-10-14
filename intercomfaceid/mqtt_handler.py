import os
import paho.mqtt.client as mqtt
import json
import time

class MQTTHandler:
    def __init__(self):
        self.mqtt_broker = os.getenv("MQTT_BROKER", "core-mosquitto")
        self.mqtt_port = 1883
        self.mqtt_username = os.getenv("MQTT_USERNAME", "mqtt")
        self.mqtt_password = os.getenv("MQTT_PASSWORD", "mqtt")

        self.bell_state_topic = "homeassistant/binary_sensor/bell_run/state"
        self.learn_face_command_topic = "homeassistant/button/learn_new_face/command"
        self.unlock_door_command_topic = "homeassistant/button/unlock_door/command"
        self.recognize_face_command_topic = "homeassistant/button/recognize_face/command"

        self.mqtt_client = mqtt.Client()
        if self.mqtt_username and self.mqtt_password:
            self.mqtt_client.username_pw_set(self.mqtt_username, self.mqtt_password)

        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_publish = self.on_publish

        self.face_recognizer = None
        self.arduino = None

        print(f"Connecting to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}")
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.mqtt_client.loop_start()

        # Wait for connection to be established
        time.sleep(2)

        # Broadcast device types after connection
        self.broadcast_device_types()

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe(self.learn_face_command_topic)
        client.subscribe(self.unlock_door_command_topic)
        client.subscribe(self.recognize_face_command_topic)

    def set_face_recognizer(self, face_recognizer):
        self.face_recognizer = face_recognizer

    def set_arduino(self, arduino):
        self.arduino = arduino

    def on_message(self, client, userdata, msg):
        print(f"Received message on topic {msg.topic}: {msg.payload.decode()}")
        if msg.topic == self.learn_face_command_topic:
            print("Received command to learn new face")
            if self.face_recognizer:
                self.face_recognizer.learn_new_face()
            else:
                print("Face recognizer not set")
        elif msg.topic == self.unlock_door_command_topic:
            print("Received command to unlock door")
            if self.arduino:
                self.arduino.unlock()
            else:
                print("Arduino handler not set")
        elif msg.topic == self.recognize_face_command_topic:
            print("Received command to recognize face")
            if self.face_recognizer:
                self.face_recognizer.captureFace()
            else:
                print("Face recognizer not set")

    def on_publish(self, client, userdata, mid):
        print(f"Message {mid} published successfully")

    def publish_bell_state(self, state):
        result = self.mqtt_client.publish(self.bell_state_topic, state)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"Published bell state: {state}")
        else:
            print(f"Failed to publish bell state: {state}")

    def broadcast_device_types(self):
        print("Broadcasting device types...")
        
        devices = [
            {
                "name": "Learn New Face",
                "unique_id": "learn_new_face_button",
                "command_topic": self.learn_face_command_topic,
                "device_class": "button",
                "payload_press": "PRESS",
                "state_topic": self.learn_face_command_topic
            },
            {
                "name": "Unlock Door",
                "unique_id": "unlock_door_button",
                "command_topic": self.unlock_door_command_topic,
                "device_class": "button",
                "payload_press": "PRESS",
                "state_topic": self.unlock_door_command_topic
            },
            {
                "name": "Recognize Face",
                "unique_id": "recognize_face_button",
                "command_topic": self.recognize_face_command_topic,
                "device_class": "button",
                "payload_press": "PRESS",
                "state_topic": self.recognize_face_command_topic
            },
            {
                "name": "Bell Run",
                "unique_id": "bell_run_sensor",
                "state_topic": self.bell_state_topic,
                "device_class": "motion",
                "payload_on": "ON",
                "payload_off": "OFF"
            }
        ]

        for device in devices:
            config_topic = f"homeassistant/{device['device_class']}/{device['unique_id']}/config"
            payload = json.dumps(device)
            result = self.mqtt_client.publish(config_topic, payload, retain=True)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"Published config for {device['name']}")
            else:
                print(f"Failed to publish config for {device['name']}")

    def process_messages(self):
        # This method can be called periodically to process MQTT messages
        # The actual message processing happens in the background due to loop_start()
        pass