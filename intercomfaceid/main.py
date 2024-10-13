import os
import time
import json
import requests

# Home Assistant Supervisor API endpoint
HASS_API_URL = "http://supervisor/core/api"

# Get the Home Assistant Supervisor token from the environment
HASS_TOKEN = os.getenv("SUPERVISOR_TOKEN")

# Headers for API requests
HEADERS = {
    "Authorization": f"Bearer {HASS_TOKEN}",
    "Content-Type": "application/json",
}
print(HEADERS)
print(HASS_API_URL)
print(HASS_TOKEN)
# Function to unlock the door
def unlock_door():
    print("Unlocking the door...")
    # Implement your actual door unlocking logic here
    time.sleep(1)
    print("Door unlocked!")

# Function to save Face ID
def save_faceid(face_data):
    print(f"Saving face ID: {face_data}")
    # Implement logic to save face ID
    time.sleep(1)
    print("Face ID saved!")

# Function to trigger when the bell is rung
def ring_bell():
    print("Bell rung!")
    # Update the status sensor in Home Assistant to "on"
    update_doorbell_sensor("on")
    time.sleep(2)
    update_doorbell_sensor("off")

# Function to update the doorbell status sensor in Home Assistant
def update_doorbell_sensor(state):
    print(f"Updating doorbell sensor to: {state}")
    sensor_data = {
        "state": state,
        "attributes": {
            "friendly_name": "Doorbell Status",
            "device_class": "sound"
        }
    }
    response = requests.post(
        f"{HASS_API_URL}/states/binary_sensor.doorbell_status",
        headers=HEADERS,
        data=json.dumps(sensor_data)
    )
    if response.status_code == 200:
        print("Doorbell sensor updated successfully!")
    else:
        print(f"Failed to update sensor: {response.text}")

# Function to handle service calls from Home Assistant
def handle_service_call(service):
    if service == "unlock_door":
        unlock_door()
    elif service == "save_faceid":
        face_data = {"name": "John Doe", "face_id": "face123"}  # Dummy face data
        save_faceid(face_data)

# Main loop to simulate the add-on running
def main():
    print("Intercom Face ID Add-on is running...")

    while True:
        # Simulating doorbell ring event every 30 seconds for testing
        ring_bell()
        time.sleep(30)

        # Simulate calling the services (in a real case, Home Assistant would trigger these)
        handle_service_call("unlock_door")
        time.sleep(5)  # Wait before calling the next service
        handle_service_call("save_faceid")

if __name__ == "__main__":
    main()