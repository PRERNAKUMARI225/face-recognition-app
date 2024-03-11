import face_recognition
import cv2
import pickle
import os
import datetime
import mysql.connector
from gtts import gTTS
from playsound import playsound

# Establish a connection to the MySQL database
connection = mysql.connector.connect(
    host='localhost',      # Host where your MySQL server is running
    database='mydatabase',    # Name of your database
    user='root',      # Username for MySQL
    password='A1B2C3d$'   # Password for MySQL
)

# Check if the connection is successful
if connection.is_connected():
    print('Connected to MySQL database')
else:
    print('Failed to connect to MySQL database')
    exit()

# Create a cursor object
cursor = connection.cursor()

# Function to insert face data into MySQL database
def insert_face_data(name, timestamp):
    try:
        # Insert face data into the database
        query = "INSERT INTO face_data (name, timestamp) VALUES (%s, %s)"
        values = (name, timestamp)
        cursor.execute(query, values)
        connection.commit()
        print("Inserted face data into the database successfully")

        # Speak the name of the detected person in Hindi
        tts = gTTS(text=f"{name} attendance lag gaya dhanyabaad", lang='en')
        tts.save("output.mp3")
        playsound("output.mp3")

    except mysql.connector.Error as e:
        print(f"Error inserting face data into the database: {e}")
        connection.rollback()

# Load or create known faces data
known_faces_data_file = "known_faces_data.pkl"
if os.path.exists(known_faces_data_file):
    with open(known_faces_data_file, "rb") as file:
        known_faces_data = pickle.load(file)
else:
    known_faces_data = {"encodings": [], "names": []}

# Function to add new face directly from code
def add_new_face(name, face_encoding):
    known_faces_data["encodings"].append(face_encoding)
    known_faces_data["names"].append(name)

    # Save updated known faces data
    with open(known_faces_data_file, "wb") as file:
        pickle.dump(known_faces_data, file)

    print(f"Added new face: {name}")

# Initialize variables
last_detection_time = {}

# Set a threshold for face recognition confidence
face_recognition_threshold = 0.60 # You can adjust this value according to your needs

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)  # 0 for the default webcam, you can change it according to your setup

# Main loop
while True:
    # Read video frame
    ret, frame = video_capture.read()

    # Resize frame for faster processing (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (OpenCV default) to RGB color
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for face_encoding, location in zip(face_encodings, face_locations):
        # Compare face encoding with known faces
        matches = face_recognition.compare_faces(known_faces_data["encodings"], face_encoding, tolerance=face_recognition_threshold)
        name = "Unknown"
        match_percentage = 0

        if True in matches:
            # Retrieve the name of the known face
            first_match_index = matches.index(True)
            name = known_faces_data["names"][first_match_index]

            # Calculate face recognition percentage
            face_distances = face_recognition.face_distance(known_faces_data["encodings"], face_encoding)
            match_percentage = (1 - face_distances[first_match_index]) * 100

            # Retrieve last detection time from database
            last_detected_time = last_detection_time.get(name)

            if last_detected_time is None or (datetime.datetime.now() - last_detected_time).seconds >= 900:
                if match_percentage >= (face_recognition_threshold*100):
                    # Save face data to MySQL database
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    insert_face_data(name, timestamp)
                    last_detection_time[name] = datetime.datetime.now()
                else:
                    name = "Unknown"
        else:
            # If no match found, consider it as Unknown
            name = "Unknown"

        # Display results
        top, right, bottom, left = [i * 4 for i in location]  # Scale back up face locations since we scaled them down
        color = (255, 0, 0)  # Blue color for known faces
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"{name} ({match_percentage:.2f}%)", (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Recognition', frame)

    # Check if 'a' key is pressed to add a new face
    if cv2.waitKey(1) & 0xFF == ord('a'):
        # Prompt user to enter name
        name = input("Enter name of the person: ")

        # Capture an image from the webcam
        ret, new_frame = video_capture.read()

        # Resize frame for faster processing (optional)
        small_new_frame = cv2.resize(new_frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (OpenCV default) to RGB color
        rgb_small_new_frame = small_new_frame[:, :, ::-1]

        # Find all the faces and their encodings in the current frame
        new_face_locations = face_recognition.face_locations(rgb_small_new_frame)
        new_face_encodings = face_recognition.face_encodings(small_new_frame, new_face_locations)

        if len(new_face_encodings) > 0:
            new_face_encoding = new_face_encodings[0]
            add_new_face(name, new_face_encoding)
            last_detection_time[name] = datetime.datetime.now()
        else:
            print("No face detected in the captured image")

    # Check if 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()

# Close MySQL connection
cursor.close() 
connection.close()
