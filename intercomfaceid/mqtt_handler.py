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

        self.bell_state_topic = "homeassistant/device_automation/intercom/action_single"
        self.learn_face_command_topic = "homeassistant/button/learn_new_face"
        self.unlock_door_command_topic = "homeassistant/button/unlock_door"
        self.recognize_face_command_topic = "homeassistant/button/recognize_face"
        self.face_recognition_result_topic = "homeassistant/sensor/recognized_person"

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
        client.subscribe(self.learn_face_command_topic + "/command")
        client.subscribe(self.unlock_door_command_topic + "/command")
        client.subscribe(self.recognize_face_command_topic + "/command")

    def set_face_recognizer(self, face_recognizer):
        self.face_recognizer = face_recognizer

    def set_arduino(self, arduino):
        self.arduino = arduino

    def on_message(self, client, userdata, msg):
        print(f"Received message on topic {msg.topic}: {msg.payload.decode()}")
        if msg.topic == self.learn_face_command_topic + "/command":
            print("Received command to learn new face")
            if self.face_recognizer:
                self.face_recognizer.learn_new_face()
            else:
                print("Face recognizer not set")
        elif msg.topic == self.unlock_door_command_topic + "/command":
            print("Received command to unlock door")
            if self.arduino:
                self.arduino.unlock()
            else:
                print("Arduino handler not set")
        elif msg.topic == self.recognize_face_command_topic + "/command":
            print("Received command to recognize face")
            if self.face_recognizer:
                name = self.face_recognizer.captureFace()
            else:
                print("Face recognizer not set")

    def publish_face_recognized(self, person_name):
        self.mqtt_client.publish(self.face_recognition_result_topic + "/state", person)

    def on_publish(self, client, userdata, mid):
        print(f"Message {mid} published successfully")

    def publish_bell_state(self):
        result = self.mqtt_client.publish(self.bell_state_topic + "/state", "single")
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"Published bell state")
        else:
            print(f"Failed to publish bell state")

    def broadcast_device_types(self):
        print("Broadcasting device types...")
        
        devices = [
            {
                "name": "Learn New Face",
                "unique_id": "learn_new_face_button",
                "command_topic": self.learn_face_command_topic + "/command",
                "device": {
                    "identifiers": ["intercom"],
                    "name": "Intercom",
                    "model": "TCS Hack",
                    "manufacturer": "TCS Daniel",
                    "sw_version": "1.0"
                }
            },
            {
                "name": "Unlock Door",
                "unique_id": "unlock_door_button",
                "command_topic": self.unlock_door_command_topic + "/command",
                "device": {
                    "identifiers": ["intercom"],
                    "name": "Intercom",
                    "model": "TCS Hack",
                    "manufacturer": "TCS Daniel",
                    "sw_version": "1.X"
                }
            },
            {
                "name": "Recognize Face",
                "unique_id": "recognize_face_button",
                "command_topic": self.recognize_face_command_topic + "/command",
                "device": {
                    "identifiers": ["intercom"],
                    "name": "Intercom",
                    "model": "TCS Hack",
                    "manufacturer": "TCS Daniel",
                    "sw_version": "1.X"
                }
            },
            {
                "name": "Bell Ring",
                "unique_id": "bell_ring_sensor",
                "topic": self.bell_state_topic+"/state",
                "automation_type":"trigger",
                "device": {
                    "identifiers": [
                    "intercom"
                    ],
                    "name": "Intercom",
                    "model": "TCS Hack",
                    "manufacturer": "TCS Daniel",
                    "sw_version": "1.0"
                },
                "type": "action",
                "subtype": "single",
                "payload": "single"
            },
            {
                "name": "Recognized Person",
                "unique_id": "recognized_person_sensor",
                "state_topic": self.face_recognition_result_topic+"/state",
                "expire_after": 5,
                "device": {
                    "identifiers": [
                    "intercom"
                    ],
                    "name": "Intercom",
                    "model": "TCS Hack",
                    "manufacturer": "TCS Daniel",
                    "sw_version": "1.0"
                },
                "type": "sensor"
            }
        ]

        for device in devices:
            if "command_topic" in device:
                topic_base = device["command_topic"].rsplit('/', 1)[0]
            elif "topic" in device:
                topic_base = device["topic"].rsplit('/', 1)[0]
            elif "state_topic" in device:
                topic_base = device["state_topic"].rsplit('/', 1)[0]
            else:
                continue

            config_topic = f"{topic_base}/config"
            payload = json.dumps(device)
            result = self.mqtt_client.publish(config_topic, payload, retain=True)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"Published config for {device['name']}")
            else:
                print(f"Failed to publish config for {device['name']}")

    def process_messages(self):
        pass