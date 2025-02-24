from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
import io

app = FastAPI()

# Enable CORS for localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change port if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



model = YOLO("yolov8n-face.pt")  # Use a custom-trained face model

# Create a rembg session
session = new_session()

@app.post("/crop-face")
async def crop_face(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run YOLOv8 Face Detection
        results = model(image)
        
        if len(results[0].boxes) == 0:
            return Response(content="No face detected", status_code=400)

        # Extract first detected face
        x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[0])
         # Add extra padding for passport photo framing
        padding_x = int((x2 - x1) * 0.8)  # 25% extra space on each side
        padding_y = int((y2 - y1) * 0.4)   # 40% extra space on top and bottom

        # Ensure coordinates do not go outside the image
        h, w, _ = image.shape
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(w, x2 + padding_x)
        y2 = min(h, y2 + padding_y)

        # Crop the face with extra space
        cropped_face = image[y1:y2, x1:x2]

        # Convert to PIL Image and Resize to Passport Size
        pil_image = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))

        pil_image_no_bg = remove(pil_image, session=session)

        bg_color = (173, 216, 230)  # Light blue color
        final_image = Image.new("RGB", (600, 600), bg_color)

        # Resize cropped face while keeping aspect ratio
        face_resized = pil_image_no_bg.resize((600, 550), Image.LANCZOS)

        # Calculate position to **center the face**
        paste_x = (600 - face_resized.width) // 2
        paste_y = 600 - face_resized.height  # Ensures face stays in the bottom

        # Paste onto the light blue background
        final_image.paste(face_resized, (paste_x, paste_y), mask=face_resized.split()[3])

        react_size = (193, 193)  # Rounded for better scaling
        resized_for_react = final_image.resize(react_size, Image.Resampling.LANCZOS)


        # Convert Image to Bytes for Response
        img_io = io.BytesIO()
        resized_for_react.save(img_io, format="JPEG", quality=95)
        img_io.seek(0)

        return Response(content=img_io.getvalue(), media_type="image/jpeg", status_code=200)

    except Exception as e:
        print(e)
        return Response(content=f"Error: {str(e)}", status_code=500)



@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    try:
        # Read input image
        input_image = Image.open(io.BytesIO(await file.read()))
        input_array = np.array(input_image)

        # Remove background
        output_array = remove.remove(input_array)
        output_image = Image.fromarray(output_array)

        # Convert image to bytes
        img_io = io.BytesIO()
        output_image.save(img_io, format="PNG")
        img_io.seek(0)

        return Response(content=img_io.getvalue(), media_type="image/png", status_code=200)
   
    except Exception as e:
        return Response(content=f"Error: {str(e)}", status_code=500)


@app.get("/")
async def check():
    return "hii"