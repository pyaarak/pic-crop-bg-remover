from fastapi import FastAPI, File, UploadFile, Response, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from ultralytics import YOLO
import numpy as np
import io
import torch
from mtcnn import MTCNN
import stripe
from pydantic import BaseModel

app = FastAPI()

# Set your Stripe API key
stripe.api_key = "sk_test_51Qr89wKNAPxQ8Uyg1W9MHkdkQWdQE0DMTUKxTzGmdVobyHApaUr5rPoyFQZ3lULut3swkD8dysoIe34n3LoV4VSI00HpJn9d1A"  # Replace with your Stripe secret key
webhook_secret = "whsec_XXXXXXXXXXXXXXXXXXXXXXXX"  # Replace with your Stripe webhook secret


# Enable CORS for localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change port if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



model = YOLO("yolov8n-face.pt")  # Use a custom-trained face model
# modelRemove = YOLO("yolov8n-seg.pt")  # Use a custom-trained face model


# Create a rembg session
session = new_session(model_name="u2net_human_seg")


# Load YOLO models
modelRemove = YOLO("yolov8n-seg.pt")  # Background removal model
modelFace = YOLO("yolov8n-face.pt")   # Face detection model

# Load FaceNet's MTCNN detector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_detector = MTCNN()


def apply_studio_editing(image: Image):
    # Enhance brightness and contrast
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)  # Increase brightness by 10%

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)  # Increase contrast by 20%

    # Sharpen the image
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.8)  # Increase sharpness

    # Reduce noise
    image = image.filter(ImageFilter.SMOOTH_MORE)

    # Adjust saturation
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.1)  # Increase saturation by 10%

    return image

def align_face(image, landmarks):
    # Align the face based on eye positions (basic alignment)
    angle = np.degrees(np.arctan2(0, 0))
    return image.rotate(angle, resample=Image.BICUBIC, expand=True)

def calculate_face_position(cropped_face, landmarks):
    """
    Calculate the position of the face and eyes to ensure they meet passport requirements.
    """
    # Get the dimensions of the cropped face
    face_height = cropped_face.height
    face_width = cropped_face.width

    # Calculate eye position (assuming landmarks are available)
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    eye_center_y = (left_eye[1] + right_eye[1]) // 2

    # Calculate head height (from top of head to chin)
    head_height = face_height

    # Ensure head height is between 25.4 mm and 35 mm
    # Convert mm to pixels (assuming 300 DPI: 1 mm = 11.81 pixels)
    min_head_height_px = int(25.4 * 11.81)
    max_head_height_px = int(35 * 11.81)

    # Resize the face to ensure head height is within the required range
    if head_height < min_head_height_px or head_height > max_head_height_px:
        scaling_factor = min_head_height_px / head_height if head_height < min_head_height_px else max_head_height_px / head_height
        new_height = int(face_height * scaling_factor)
        new_width = int(face_width * scaling_factor)
        cropped_face = cropped_face.resize((new_width, new_height), Image.LANCZOS)

    return cropped_face

def feather_edges(image):
    # Create a mask from the alpha channel
    mask = image.split()[3]

    # Apply Gaussian blur to the mask
    blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=1))

    # Paste the image onto a new background using the blurred mask
    feathered_image = Image.new("RGBA", image.size, (255, 255, 255, 0))
    feathered_image.paste(image, (0, 0), blurred_mask)

    return feathered_image


def smooth_edges(image):
    # Convert PIL image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to create a mask
    _, mask = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with smoothed edges
    smoothed_mask = np.zeros_like(mask)
    cv2.drawContours(smoothed_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, kernel)
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image_cv, image_cv, mask=smoothed_mask)

    # Convert back to PIL format
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    return result_pil

def remove_shadows_clahe(image):
    # Convert PIL image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Convert to LAB color space
    lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge the CLAHE-enhanced L channel back with the A and B channels
    lab_clahe = cv2.merge((l_clahe, a, b))

    # Convert back to BGR color space
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Convert back to PIL format
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    return result_pil

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Convert image to a NumPy array with dtype uint8
        input_array = np.array(input_image, dtype=np.uint8)

        # Detect face using MTCNN
        result = face_detector.detect_faces(input_array)
        if not result:
            return Response(content="No face detected", status_code=400)

        # Extract the first detected face and landmarks
        box = result[0]['box']
        landmarks = result[0]['keypoints']
        x1, y1, width, height = box
        x2, y2 = x1 + width, y1 + height

        # Add padding (extra space for passport photo)
        padding_x = int(width * 0.75)
        padding_y = int(height * 0.45)
        padding_y_bottom = int(height * 0.5)  # Extra padding below the face to include bod

        x1 = max(0, x1 - padding_x)
        x2 = min(input_image.width, x2 + padding_x)
        y1 = max(0, y1 - padding_y)
        y2 = min(input_image.height, y2 + padding_y_bottom)

        # Crop the face region
        cropped_face = input_image.crop((x1, y1, x2, y2))

        # Ensure face and eye positioning meet passport requirements
        cropped_face = calculate_face_position(cropped_face, landmarks)

        # Remove background using Rembg
        no_bg_image = remove(cropped_face)

         # Smooth edges to remove shadows
        smoothed_image = feather_edges(no_bg_image)

        # Resize to passport size (600x600 pixels for 2x2 inches at 300 DPI)
        face_resized = smoothed_image.resize((1200, 1200), Image.LANCZOS)

        # Create a passport-size background (white)
        white_background = (255, 255, 255)
        passport_background = Image.new("RGB", (1200, 1200), white_background)

        # Center the face on the background
        paste_x = (1200 - face_resized.width) // 2
        paste_y = (1200 - face_resized.height) // 2
        passport_background.paste(face_resized, (paste_x, paste_y), face_resized.split()[3])

        # Apply studio-like enhancements
        passport_background = apply_studio_editing(passport_background)

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        passport_background.save(img_byte_arr, format="PNG", quality=100)
        img_byte_arr.seek(0)

        return Response(content=img_byte_arr.getvalue(), media_type="image/png", status_code=200)

    except Exception as e:
        print(e)
        return Response(content=f"Error: {str(e)}", status_code=500)


@app.post("/process-image2")
async def process_image2(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Convert image to numpy array
        input_array = np.array(input_image, dtype=np.uint8)

        # Detect face using MTCNN
        boxes = face_detector.detect_faces(input_array)

        if boxes is None:
            return Response(content="No face detected", status_code=400)

         # Extract the first detected face and landmarks
        box = boxes[0]['box']
        landmarks = boxes[0]['keypoints']
        x1, y1, width, height = box
        x2, y2 = x1 + width, y1 + height

        # Add padding (extra space for passport photo)
        padding_x = int(width * 0.7)
        padding_y = int(height * 0.3)

        x1 = max(0, x1 - padding_x)
        x2 = min(input_image.width, x2 + padding_x)
        y1 = max(0, y1 - padding_y)
        y2 = min(input_image.height, y2 + padding_y)

        # Crop the face region
        cropped_face = input_image.crop((x1, y1, x2, y2))

        # Align the face
        cropped_face = align_face(cropped_face, landmarks)

        # Remove background using Rembg
        no_bg_image = remove(cropped_face)

        # Clean the alpha channel to remove black lines
        cleaned_image = clean_alpha_channel(no_bg_image)

         # Smooth edges to remove shadows
        smoothed_image = feather_edges(cleaned_image)

        # Resize to passport size (600x600 pixels for 2x2 inches at 300 DPI)
        face_resized = smoothed_image.resize((1200, 1200), Image.LANCZOS)

        # Create a passport-size background (light blue)
        light_blue = (255, 255, 255)
        passport_background = Image.new("RGB", (1200, 1200), light_blue)

        # Center the face on the background
        paste_x = (1200 - face_resized.width) // 2
        paste_y = 1200 - face_resized.height
        passport_background.paste(face_resized, (paste_x, paste_y), face_resized.split()[3])

        # Apply studio-like enhancements
        passport_background = apply_studio_editing(passport_background)

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        passport_background.save(img_byte_arr, format="PNG", quality=100)
        img_byte_arr.seek(0)

        return Response(content=img_byte_arr.getvalue(), media_type="image/png", status_code=200)

    except Exception as e:
        print(e)
        return Response(content=f"Error: {str(e)}", status_code=500)


@app.post("/process-image1")
async def process_image1(file: UploadFile = File(...)):
    try:
        # Read and load image
        image_bytes = await file.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_array = np.array(input_image)

        # Step 1: Detect Face
        face_results = modelFace(input_array)[0]

        if not face_results.boxes or len(face_results.boxes) == 0:
            return Response(content="No face detected", status_code=400)

        # Get face bounding box
        x1, y1, x2, y2 = face_results.boxes.xyxy[0].cpu().numpy()
        h, w = input_image.height, input_image.width

        # Step 2: Add Extra Padding (Ensure Full Head is Included)
        padding_x = int((x2 - x1) * 0.7)  # 30% extra width
        padding_y = int((y2 - y1) * 0.4)  # 50% extra height (more top space)

        x1 = max(0, int(x1 - padding_x))
        x2 = min(w, int(x2 + padding_x))
        y1 = max(0, int(y1 - padding_y))  # Shift face down for more top space
        y2 = min(h, int(y2 + padding_y))

        # Step 3: Crop the Face
        cropped_face = input_array[y1:y2, x1:x2]

        # Step 4: Background Removal with YOLOv8 Segmentation
        results = modelRemove(input_array)[0]
        mask = results.masks.data[0].numpy()  # Get mask
        mask = cv2.resize(mask, (w, h))

        # Convert mask to binary (Remove Noise & Small Artifacts)
        mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

        # Apply Morphological Operations (Close Holes, Refine Edges)
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # Close gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove small noise

        # Apply mask to the cropped face region
        cropped_mask = mask[y1:y2, x1:x2]
        bg_removed = cv2.bitwise_and(cropped_face, cropped_face, mask=cropped_mask)

        # Step 5: Smooth Edges (Gaussian Blur on Mask)
        blurred_mask = cv2.GaussianBlur(cropped_mask, (25, 25), 10)  # Soft blending

        # Step 6: Replace Background with Light Blue
        light_blue = (173, 216, 230)  # Light blue color (RGB)
        solid_bg = np.full_like(cropped_face, light_blue, dtype=np.uint8)

        # Blend face with background using blurred mask
        final_image = np.where(blurred_mask[:, :, None] > 100, bg_removed, solid_bg)

        # Step 7: Convert to PIL Image & Enhance Quality
        output_image = Image.fromarray(final_image)
        output_image = output_image.filter(ImageFilter.DETAIL)  # Enhance details
        output_image = ImageEnhance.Sharpness(output_image).enhance(1.2)  # Sharpening

        # Step 8: Resize to 600x550 for Passport Format
        face_resized = output_image.resize((600, 550), Image.LANCZOS)

        # Step 9: Create 600x600 Light Blue Background
        passport_background = Image.new("RGB", (600, 600), light_blue)

        # Step 10: Paste Face at Bottom (Extra Top Space)
        paste_x = (600 - face_resized.width) // 2
        paste_y = 600 - face_resized.height  # Align face to bottom
        passport_background.paste(face_resized, (paste_x, paste_y))

        # Step 11: Save as PNG
        img_byte_arr = io.BytesIO()
        passport_background.save(img_byte_arr, format="PNG", quality=95)
        img_byte_arr.seek(0)

        return Response(content=img_byte_arr.getvalue(), media_type="image/png", status_code=200)

    except Exception as e:
        print(e)
        return Response(content=f"Error: {str(e)}", status_code=500)



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

        react_size = (600, 600)  # Rounded for better scaling
        resized_for_react = final_image.resize(react_size, Image.Resampling.LANCZOS)


        # Convert Image to Bytes for Response
        img_io = io.BytesIO()
        resized_for_react.save(img_io, format="PNG", quality=95)
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
    
@app.post("/remove-bg-yolo")
async def remove_bg(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_array = np.array(input_image)

        # Run YOLOv8 segmentation
        results = modelRemove(input_array)[0]

        # Create mask and remove background
        mask = results.masks.data[0].numpy()  # Get mask
        mask = cv2.resize(mask, (input_image.width, input_image.height))
        mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask

        # Apply mask
        bg_removed = cv2.bitwise_and(input_array, input_array, mask=mask)

        # Convert back to PIL
        output_image = Image.fromarray(bg_removed)
        output_image = output_image.filter(ImageFilter.DETAIL)

        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format="PNG",quality=95)
        img_byte_arr.seek(0)

        return Response(content=img_byte_arr.getvalue(), media_type="image/png", status_code=200)
   
    except Exception as e:
        print (e)
        return Response(content=f"Error: {str(e)}", status_code=500)

# Pydantic model for creating a payment intent
class CreatePaymentIntentRequest(BaseModel):
    amount: int  # Amount in the smallest currency unit (e.g., cents for USD)
    currency: str = "usd"

# Pydantic model for confirming a payment
class ConfirmPaymentRequest(BaseModel):
    payment_intent_id: str  # Payment Intent ID from the client
    payment_method: str  # Payment Method ID from the client

@app.post("/create-payment-intent")
async def create_payment_intent(payment_request: CreatePaymentIntentRequest):
    """
    Create a Stripe Payment Intent.
    """
    try:
        # Create a Payment Intent
        intent = stripe.PaymentIntent.create(
            amount=payment_request.amount,
            currency=payment_request.currency,
        )
        return {"client_secret": intent["client_secret"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/confirm-payment")
async def confirm_payment(confirm_request: ConfirmPaymentRequest):
    """
    Confirm a Stripe Payment Intent.
    """
    try:
        # Confirm the Payment Intent
        intent = stripe.PaymentIntent.confirm(
            confirm_request.payment_intent_id,
            payment_method=confirm_request.payment_method,
        )
        return {"status": intent["status"], "payment_intent_id": intent["id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/")
async def check():
    return "hii"