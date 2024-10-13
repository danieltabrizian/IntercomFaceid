import os
import paho.mqtt.client as mqtt
import time
import json

# MQTT broker address and credentials
mqtt_broker = os.getenv("MQTT_BROKER", "localhost")
mqtt_username = os.getenv("MQTT_USERNAME", "mqtt")
mqtt_password = os.getenv("MQTT_PASSWORD", "mqtt")

# Initialize MQTT client
mqtt_client = mqtt.Client()

# Set MQTT credentials if needed
if mqtt_username and mqtt_password:
    mqtt_client.username_pw_set(mqtt_username, mqtt_password)

# Connect to the MQTT broker
mqtt_client.connect(mqtt_broker, 1883, 60)

# Global states
bell_running = False

# MQTT topics for states and commands
bell_state_topic = "homeassistant/binary_sensor/bell_run/state"
learn_face_command_topic = "homeassistant/switch/learn_new_face/set"
unlock_door_command_topic = "homeassistant/switch/unlock_door/set"

# Function to publish MQTT discovery messages
def publish_discovery():
    # Binary Sensor for Bell Ring
    bell_sensor_config = {
        "name": "Bell Run",
        "device_class": "sound",
        "state_topic": bell_state_topic,
        "unique_id": "bell_run_sensor",
        "payload_on": "ON",
        "payload_off": "OFF"
    }
    mqtt_client.publish("homeassistant/binary_sensor/bell_run/config", json.dumps(bell_sensor_config))

    # Switch for "Learn New Face"
    learn_face_switch_config = {
        "name": "Learn New Face",
        "command_topic": learn_face_command_topic,
        "unique_id": "learn_new_face_switch"
    }
    mqtt_client.publish("homeassistant/switch/learn_new_face/config", json.dumps(learn_face_switch_config))

    # Switch for "Unlock Door"
    unlock_door_switch_config = {
        "name": "Unlock Door",
        "command_topic": unlock_door_command_topic,
        "unique_id": "unlock_door_switch"
    }
    mqtt_client.publish("homeassistant/switch/unlock_door/config", json.dumps(unlock_door_switch_config))

# Function to publish the state of the bell sensor
def publish_bell_state(state):
    mqtt_client.publish(bell_state_topic, state)
    print(f"Published bell state: {state}")

# Function to handle incoming messages (button presses)
def on_message(client, userdata, msg):
    if msg.topic == learn_face_command_topic:
        if msg.payload.decode() == "ON":
            learn_new_face()

    elif msg.topic == unlock_door_command_topic:
        if msg.payload.decode() == "ON":
            unlock_door()

# Function to simulate unlocking the door
def unlock_door():
    print("Unlocking the door...")
    # Add your actual unlocking logic here
    time.sleep(2)
    print("Door unlocked!")

# Function to simulate learning a new face
def learn_new_face():
    print("Learning new face...")
    # Add your actual face recognition logic here
    time.sleep(3)
    print("New face learned!")

# Function to simulate the bell ringing
def ring_bell():
    global bell_running
    print("Bell rung!")
    bell_running = True
    publish_bell_state("ON")
    time.sleep(2)  # Simulate the bell ringing for 2 seconds
    bell_running = False
    publish_bell_state("OFF")

# Main function
def main():
    # Subscribe to command topics for switches
    mqtt_client.subscribe(learn_face_command_topic)
    mqtt_client.subscribe(unlock_door_command_topic)

    # Set the callback function for when messages are received
    mqtt_client.on_message = on_message

    # Start MQTT client loop in a background thread
    mqtt_client.loop_start()

    # Publish the discovery messages to set up the entities
    publish_discovery()

    # Main loop simulating the bell ringing every 30 seconds
    while True:
        ring_bell()
        time.sleep(30)  # Wait 30 seconds before ringing the bell again

if __name__ == "__main__":
    main()