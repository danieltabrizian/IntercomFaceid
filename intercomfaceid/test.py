import cv2
import numpy as np
import insightface

# Load the InsightFace model (use 'arcface' for embeddings)
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0)

# Load an image
img = cv2.imread('face_image.jpeg')

# Detect faces and generate embeddings
faces = model.get(img)
for face in faces:
    print("Embedding for face:", face.embedding)