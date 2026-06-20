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

# The TCS bus carries a lot of traffic. 4-digit (or shorter) codes are heartbeat
# / noise and are dropped entirely (not logged) by the Arduino handler.
#
# DOORBELL_CODES: the code(s) that mean "someone is ringing THIS door" — the only
# ones that should trigger a snapshot + face recognition. Filled in once we
# identify it from the (now decluttered) activity log. Empty = recognition idle.
# The doorbell button on THIS unit sends 'call:0C594F80'. That's the only code
# that should trigger a snapshot + recognition. Everything else on the bus —
# other units' calls, and the unlock echo 1C594F80 the intercom emits when the
# door opens — is ignored (ignoring the echo prevents an unlock->recognize->unlock
# feedback loop).
DOORBELL_CODES = {"0C594F80"}


def _signal_code(command):
    """Extract the code portion of a 'call:XXXX' / 'Received HEX: XXXX' line."""
    if command.startswith("call:"):
        return command[len("call:"):].strip()
    if command.startswith("Received HEX:"):
        return command[len("Received HEX:"):].strip()
    return None


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
        # Read frames on demand from motionEye's MJPEG stream (the USB capturer).
        # Use the host IP, NOT homeassistant.local — resolving the .local name
        # inside the add-on container hits a ~10s unicast-DNS timeout before
        # falling back to mDNS, which dominated the bell→recognition latency.
        stream_manager = StreamManager("http://192.168.2.45:9081", autostart=False)
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
            command = arduino.read_command()  # 4-digit noise already dropped upstream
            code = _signal_code(command)
            if code is not None:
                # Only our doorbell triggers a snapshot + recognition. Everything
                # else on the bus (other units' calls, the 1C594F80 unlock echo) is
                # ignored — which also prevents the unlock->recognize->unlock loop.
                if code in DOORBELL_CODES:
                    logging.info(f"Doorbell: {command}")
                    if enable_mqtt:
                        try:
                            mqtt_client.publish_bell_state()
                        except Exception as e:
                            logging.error(f"Error publishing bell state: {e}")
                    if enable_face_recognition:
                        try:
                            face_recognizer.captureFace(run_recognition=True)
                        except Exception as e:
                            logging.error(f"Error during face capture: {e}")
                else:
                    logging.debug(f"Ignored signal: {command}")
            elif command == "unlock":
                logging.info("Received unlock command")

        time.sleep(1)

if __name__ == "__main__":
    main()
