import cv2
import numpy as np
import threading
import time
import logging
import requests
from requests.exceptions import RequestException
import queue

class StreamManager:
    def __init__(self, stream_url, max_retry_attempts=3, retry_delay=5):
        self.stream_url = stream_url
        self.current_frame = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.is_capturing = False
        self.lock = threading.Lock()
        self.capture_thread = None
        self.stream = None
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay = retry_delay
        self.watchdog_thread = None
        self.frame_queue = queue.Queue(maxsize=10)

        # Start the stream immediately upon initialization
        self.start_video_stream()
        self.start_watchdog()

    def start_video_stream(self):
        with self.lock:
            if not self.is_capturing:
                for attempt in range(self.max_retry_attempts):
                    try:
                        self.stream = requests.get(self.stream_url, stream=True, timeout=10)
                        if self.stream.status_code != 200:
                            raise Exception(f"Failed to connect to stream. Status code: {self.stream.status_code}")
                        
                        self.capture_thread = threading.Thread(target=self._capture_stream)
                        self.capture_thread.daemon = True
                        self.is_capturing = True
                        self.capture_thread.start()
                        logging.info(f"MJPEG stream capture started successfully. URL: {self.stream_url}")
                        return True
                    except RequestException as e:
                        logging.error(f"Network error when starting video stream (attempt {attempt+1}/{self.max_retry_attempts}): {str(e)}")
                    except Exception as e:
                        logging.error(f"Unexpected error when starting video stream (attempt {attempt+1}/{self.max_retry_attempts}): {str(e)}")
                    
                    if attempt < self.max_retry_attempts - 1:
                        logging.info(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                
                self.is_capturing = False
                self.capture_thread = None
                logging.error("Failed to start video stream after multiple attempts.")
                return False
            else:
                logging.info("Video stream is already running.")
                return True

    def _capture_stream(self):
        bytes_buffer = bytes()
        frame_counter = 0
        start_time = time.time()  # Record the start time for frame rate calculation

        while self.is_capturing:
            try:
                chunk = self.stream.raw.read(1024)
                if not chunk:
                    raise Exception("No data received from stream.")
                bytes_buffer += chunk

                # Search for JPEG boundaries
                while True:
                    a = bytes_buffer.find(b'\xff\xd8')
                    b = bytes_buffer.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = bytes_buffer[a:b + 2]
                        bytes_buffer = bytes_buffer[b + 2:]  # Keep any remaining bytes
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self.lock:
                                self.current_frame = frame
                                self.last_frame_time = time.time()
                                self.frame_count += 1
                                frame_counter += 1  # Increment the frame counter
                                try:
                                    # Attempt to put the frame in the queue
                                    self.frame_queue.put(frame, block=False)
                                except queue.Full:
                                    try:
                                        self.frame_queue.get_nowait()  # Remove the oldest frame
                                        self.frame_queue.put(frame, block=False)  # Add the new frame
                                    except queue.Empty:
                                        pass  # This should not happen as we're managing the size
                            break  # Exit the while loop to start capturing the next frame
                    else:
                        # Break out if we don't have a full frame yet
                        break

                # Log frame rate every 5 seconds
                current_time = time.time()
                if current_time - start_time >= 5.0:
                    fps = frame_counter / 5.0  # Calculate frames per second
                    logging.info(f"Current frame rate: {fps:.2f} FPS")
                    start_time = current_time  # Reset start time
                    frame_counter = 0  # Reset frame counter

            except Exception as e:
                logging.error(f"Error in stream capture: {str(e)}")
                # Only restart if the error is severe (not a queue.Full error)
                if not isinstance(e, queue.Full):
                    self.restart_stream()  # Attempt to restart on error
                time.sleep(0.1)  # Wait a bit before trying again

        if self.stream:
            self.stream.close()
        logging.info("MJPEG stream capture stopped.")


    def get_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None
        bytes_buffer = bytes()
        consecutive_errors = 0
        while self.is_capturing:
            try:
                chunk = self.stream.raw.read(1024)
                if not chunk:
                    raise Exception("No data received from stream.")
                
                bytes_buffer += chunk
                a = bytes_buffer.find(b'\xff\xd8')
                b = bytes_buffer.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_buffer[a:b+2]
                    bytes_buffer = bytes_buffer[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.lock:
                            self.current_frame = frame
                            self.last_frame_time = time.time()
                            self.frame_count += 1
                            if self.frame_count % 100 == 0:
                                logging.info(f"Received {self.frame_count} frames")
                        consecutive_errors = 0
                    else:
                        logging.warning("Received invalid frame data")
                        consecutive_errors += 1
            except Exception as e:
                logging.error(f"Error in stream capture: {str(e)}")
                consecutive_errors += 1
            
            if consecutive_errors >= 5:
                logging.error("Too many consecutive errors. Restarting stream...")
                self.restart_stream()
                consecutive_errors = 0
            
            time.sleep(0.01)  # Small delay to prevent CPU overuse in case of rapid errors

        if self.stream:
            self.stream.close()
        logging.info("MJPEG stream capture stopped.")

    def restart_stream(self):
        logging.info("Attempting to restart the video stream...")
        self.stop_video_stream()
        time.sleep(1)
        self.start_video_stream()

    def stop_video_stream(self):
        with self.lock:
            self.is_capturing = False
            if self.capture_thread:
                self.capture_thread.join(timeout=5)
                if self.capture_thread.is_alive():
                    logging.warning("Capture thread did not terminate gracefully.")
                self.capture_thread = None
            if self.stream:
                self.stream.close()
            self.current_frame = None
            logging.info("MJPEG stream capture stopped.")

    def start_watchdog(self):
        self.watchdog_thread = threading.Thread(target=self._watchdog)
        self.watchdog_thread.daemon = True
        self.watchdog_thread.start()

    def _watchdog(self):
        while True:
            time.sleep(10)  # Check every 10 seconds
            if self.is_capturing and time.time() - self.last_frame_time > 30:
                logging.warning("No frames received for 30 seconds. Restarting stream...")
                self.restart_stream()


    def __del__(self):
        self.stop_video_stream()
        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=5)