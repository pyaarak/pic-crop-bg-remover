from fastapi import FastAPI, File, UploadFile, Response, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
import cv2
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import io
import torch
from mtcnn import MTCNN
import stripe
from pydantic import BaseModel
import zipfile
from pillow_heif import register_heif_opener
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import insightface
from insightface.app import FaceAnalysis
from scipy.ndimage import gaussian_filter
import sys
sys.path.append("RobustVideoMatting\model")
from torchvision.transforms import ToTensor
from RobustVideoMatting.model.model import MattingNetwork
import torchvision.transforms as transforms
from MODNet.src.models.modnet import MODNet
import torch.nn as nn
from skimage import morphology
from scipy.ndimage import distance_transform_edt

modelFace = FaceAnalysis(name='buffalo_l')
modelFace.prepare(ctx_id=0, det_size=(640, 640))

modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)
modnet.load_state_dict(torch.load('modnet.ckpt', map_location=torch.device('cpu')))
modnet.eval()

modelMat = MattingNetwork('resnet50')
modelMat.load_state_dict(torch.load('rvm_resnet50.pth', map_location='cpu'))
modelMat.eval()
modelMat.to('cpu')

model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=23, num_grow_ch=32)

esrgan_model = RealESRGANer(
    scale=2,
    model_path='Realtrained.pth',
    model=model,
    pre_pad=0,
    half=False,
    device='cpu'  
)

register_heif_opener()
app = FastAPI()

gfpgan_model = GFPGANer(
    model_path='gftrained.pth',
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    device='cpu',
    bg_upsampler=None,
)

stripe.api_key = "sk_test_51Qr89wKNAPxQ8Uyg1W9MHkdkQWdQE0DMTUKxTimazGmdVobyHApaUr5rPoyFQZ3lULut3swkD8dysoIe34n3LoV4VSI00HpJn9d1A" 



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# model = YOLO("yolov8n-face.pt")  # Use a custom-trained face model
# # modelRemove = YOLO("yolov8n-seg.pt")  # Use a custom-trained face model


# # Create a rembg session
# session = new_session(model_name="u2net_human_seg")


# Load YOLO models
# modelRemove = YOLO("yolov8n-seg.pt")  # Background removal model
# modelFace = YOLO("yolov8n-face.pt")   # Face detection model

# Load FaceNet's MTCNN detector
device = torch.device("cpu")
face_detector = MTCNN()

def get_uk_passport_crop(img, face, final_size=1200,
                         target_head_ratio=0.69, eye_level_ratio=0.43,
                         min_side_padding_ratio=0.07, max_side_padding_ratio=0.1,
                         bottom_extra_ratio=0.20):
    """
    Generate a consistent UK biometric crop from SCRFD-detected face.
    - Ensures head is ~69% of image height
    - Eyes ~43% from top
    - Adds shoulder/body space if possible
    - Dynamically adapts to tight framing or excess space
    """
    h, w = img.shape[:2]

    bbox = face.bbox.astype(int)
    kps = face.kps

    face_x1, face_y1, face_x2, face_y2 = bbox
    face_w = face_x2 - face_x1
    face_h = face_y2 - face_y1

    # Estimate top of head from bbox (20% above)
    top_of_head_y = face_y1 - int(0.2 * face_h)
    top_of_head_y = max(0, top_of_head_y)

    chin_y = face_y2
    actual_head_height = chin_y - top_of_head_y

    # Get eye Y position
    eye_y = int((kps[0][1] + kps[1][1]) / 2)

    desired_head_px = int(final_size * target_head_ratio)
    scale = desired_head_px / actual_head_height

    # Initial crop dimensions
    crop_h = final_size / scale
    crop_w = crop_h

    # Horizontal padding based on face position and space
    face_cx = (face_x1 + face_x2) // 2
    available_left = face_cx
    available_right = w - face_cx
    min_space = min(available_left, available_right)
    face_half_width = face_w / 2
    max_pad_ratio = min(max_side_padding_ratio, min_space / face_half_width)
    side_padding_ratio = min(max_pad_ratio, max_side_padding_ratio)
    side_padding_ratio = max(side_padding_ratio, min_side_padding_ratio)

    crop_w *= (1 + side_padding_ratio * 2)
    crop_h *= (1 + bottom_extra_ratio)

    # Compute Y (vertical) crop origin using eye alignment
    eye_target_y = eye_level_ratio * final_size
    crop_y1 = eye_y - (eye_target_y / final_size) * crop_h
    crop_y2 = crop_y1 + crop_h

    # Compute X (horizontal) crop origin using center alignment
    crop_x1 = face_cx - crop_w / 2
    crop_x2 = crop_x1 + crop_w

    # Clamp to image
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(w, crop_x1 + crop_w)
    crop_y2 = min(h, crop_y1 + crop_h)

    # Adjust again if image edge prevents full crop
    if crop_x2 - crop_x1 < crop_w:
        crop_x1 = max(0, crop_x2 - crop_w)
    if crop_y2 - crop_y1 < crop_h:
        crop_y1 = max(0, crop_y2 - crop_h)

    # Final crop box as integers
    x1, y1, x2, y2 = map(int, [crop_x1, crop_y1, crop_x2, crop_y2])
    cropped = img[y1:y2, x1:x2]

    # Resize to output size
    output = cv2.resize(cropped, (final_size, final_size), interpolation=cv2.INTER_LANCZOS4)
    return output


LAYOUTS = {
    "Printable_3.5x5": {
        "canvas_size": (2100, 3000),  # 2 images, 1 per row (1200, 2400)
        "image_positions": [(450, 295), (450, 1505)],
    },
    "Printable_6x4": {
        "canvas_size": (3600, 2400),  # 6 images, 3 per row
        "image_positions": [
            (0, 0), (1200, 0), (2400, 0),
            (0, 1200), (1200, 1200), (2400, 1200),
        ],
    },
    "Printable_7X5": {
        "canvas_size": (4200, 3000),  # 6 images, 3 per row (3600, 2400) (594 / 2, 598 /2) (297, 299)
        "image_positions": [
            (297, 299), (1499, 299), (2701, 299),
            (297, 1501), (1499, 1501), (2701, 1501),
        ],
    },
    "Printable_A4": {
        "canvas_size": (7016, 4961),  # 20 images, 5 per row (6000, 4800) (1016 / 2, 161) (1008 / 2, 155 / 2) (504, 77)
        "image_positions": [
            (504, 77), (1706, 77), (2908, 77), (4110, 77), (5312, 77),
            (504, 1279), (1706, 1279), (2908, 1279), (4110, 1279), (5312, 1279),
            (504, 2481), (1706, 2481), (2908, 2481), (4110, 2481), (5312, 2481),
            (504, 3683), (1706, 3683), (2908, 3683), (4110, 3683), (5312, 3683),
        ],
    },
    "Printable_A5": {
        "canvas_size": (4961, 3496),  # 8 images, 4 per row (4800, 2400) (+161, +1096) (155 /2, 1094/2)
        "image_positions": [
            (77, 547), (1279, 547), (2481, 547), (3683, 547),
            (77, 1749), (1279, 1749), (2481, 1749), (3683, 1749),
        ],
    },
    "Printable_legal": {
        "canvas_size": (8400, 5100),  # 28 images, 7 per row (8400, 4800) (0, 300) ()
        "image_positions": [
            (0, 146), (1200, 146), (2400, 146), (3600, 146), (4800, 146), (6000, 146), (7200, 146),
            (0, 1348), (1200, 1348), (2400, 1348), (3600, 1348), (4800, 1348), (6000, 1348), (7200, 1348),
            (0, 2550), (1200, 2550), (2400, 2550), (3600, 2550), (4800, 2550), (6000, 2550), (7200, 2550),
            (0, 3752), (1200, 3752), (2400, 3752), (3600, 3752), (4800, 3752), (6000, 3752), (7200, 3752),
        ],
    },
    "Printable_letter": {
        "canvas_size": (6600, 5100),  # 20 images, 5 per row (6000, 4800) (592 / 2, 292)
        "image_positions": [
            (296, 146), (1498, 146), (2700, 146), (3902, 146), (5104, 146),
            (296, 1348), (1498, 1348), (2700, 1348), (3902, 1348), (5104, 1348),
            (296, 2550), (1498, 2550), (2700, 2550), (3902, 2550), (5104, 2550),
            (296, 3752), (1498, 3752), (2700, 3752), (3902, 3752), (5104, 3752),
        ],
    },
}

def create_printable_sheet(images, layout):
    canvas_size = layout["canvas_size"]
    positions = layout["image_positions"]
    sheet = Image.new("RGB", canvas_size, (207, 207, 209))
    
    for img, pos in zip(images, positions):
        sheet.paste(img, pos)
    
    return sheet

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        passport_background = img.resize((1198, 1198), Image.LANCZOS)
        passport_background_1200 = img.resize((1200, 1200), Image.LANCZOS)

        printable_sheets = {}
        for layout_name, layout in LAYOUTS.items():
            images = [passport_background] * len(layout["image_positions"])
            printable_sheets[layout_name] = create_printable_sheet(images, layout)

        single_hd_image = passport_background_1200
        single_hd_compressed = passport_background_1200.copy()
        
        single_hd_compressed_byte_arr = io.BytesIO()
        single_hd_compressed.save(single_hd_compressed_byte_arr, format="PNG", quality=50, optimize=True, progressive=True)
        single_hd_compressed_byte_arr.seek(0)

        img_byte_arrs = {}
        for layout_name, sheet in printable_sheets.items():
            img_byte_arr = io.BytesIO()
            sheet.save(img_byte_arr, format="PNG", quality=100)
            img_byte_arr.seek(0)
            img_byte_arrs[layout_name] = img_byte_arr.getvalue()

        single_hd_byte_arr = io.BytesIO()
        single_hd_image.save(single_hd_byte_arr, format="PNG",quality=100)
        single_hd_byte_arr.seek(0)
        img_byte_arrs["single_hd_image"] = single_hd_byte_arr.getvalue()
        img_byte_arrs["single_hd_compressed"] = single_hd_compressed_byte_arr.getvalue()

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for name, img_bytes in img_byte_arrs.items():
                zip_file.writestr(f"{name}.png", img_bytes)
        zip_buffer.seek(0)

        return Response(content=zip_buffer.getvalue(), media_type="application/zip", headers={"Content-Disposition": "attachment; filename=printable_sheets.zip"})
    
    except Exception as e:
        print(e)
        return Response(content=f"Error: {str(e)}", status_code=500)


def force_shirt_region_smart(alpha_np, strength=0.95, min_existing_alpha=0.4):
    h, w = alpha_np.shape
    bottom_mask = np.zeros_like(alpha_np)
    bottom_mask[int(h * 0.4):, :] = 1 

    boosted_alpha = np.where(
        (bottom_mask == 1) & (alpha_np > min_existing_alpha),
        np.maximum(alpha_np, strength),
        alpha_np
    )
    return boosted_alpha

def boost_shirt_alpha(alpha_np, strength=0.95):
    h, w = alpha_np.shape
    bottom_mask = np.zeros_like(alpha_np)
    bottom_mask[int(h * 0.4):, :] = 1 

    boosted = np.where(bottom_mask == 1, 
                       np.maximum(alpha_np, strength), 
                       alpha_np)
    return boosted

def repair_alpha_matte(alpha_np):
    alpha_uint8 = (alpha_np * 255).astype(np.uint8)
    _, binary = cv2.threshold(alpha_uint8, 40, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    blurred = cv2.GaussianBlur(dilated, (7, 7), 0)
    repaired_alpha = blurred.astype(np.float32) / 255.0
    repaired_alpha = np.clip(repaired_alpha, 0, 1)

    return repaired_alpha

def advanced_clean_rvm_alpha(pha, threshold=0.85, min_area=1500, feather_radius=25, fill_holes_radius=25):
    pha = np.clip(pha * 1.2, 0, 1)
    binary = (pha > threshold).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels): 
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((fill_holes_radius, fill_holes_radius), np.uint8))
    cleaned = cv2.GaussianBlur(cleaned, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    cleaned = cv2.erode(cleaned, kernel, iterations=1)
    cleaned = cleaned.astype(np.float32) / 255.0
    return cleaned



def remove_bg_rvm_cpu(image: Image.Image, model=modelMat):
    frame = ToTensor()(image).unsqueeze(0).to("cpu")
    rec = [None] * 4
    downsample_ratio = torch.tensor([[1]], dtype=torch.float32).to("cpu")
    with torch.no_grad():
        fgr, pha, *_ = model(frame, *rec, downsample_ratio)

    fgr = fgr[0].permute(1, 2, 0).cpu().numpy()
    pha = pha[0, 0].cpu().numpy()
    pha = clean_alpha_mask(pha)
    pha = repair_alpha_matte(pha)
    pha = boost_shirt_alpha(pha, strength=0.95)
    pha = final_small_patch_cleaner(pha, min_area=500)
    comp = fgr * pha[..., None] + (1 - pha[..., None]) * 1
    comp_img = Image.fromarray((comp * 255).astype(np.uint8))

    return comp_img

def final_small_patch_cleaner(alpha_np, min_area=500):
    alpha_uint8 = (alpha_np * 255).astype(np.uint8)
    _, binary = cv2.threshold(alpha_uint8, 128, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
    cleaned = cv2.GaussianBlur(cleaned, (5, 5), 0) 
    cleaned = cleaned.astype(np.float32) / 255.0
    cleaned = np.clip(cleaned, 0, 1)

    return cleaned


def clean_stray_hair_with_morph(matte, kernel_size=15, iterations=3):
    binary = (matte > 0.4).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = opened.astype(np.float32) * matte
    return cleaned

def suppress_vertical_strays(matte, min_height_ratio=0.15, width_threshold=10):
    h, w = matte.shape
    binary = (matte > 0.3).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if bh > h * min_height_ratio and bw < width_threshold:
            matte[labels == i] = 0
    return matte



def soften_matte_edges(matte, sigma=1.5):
    return gaussian_filter(matte, sigma=sigma)

def extract_main_head_mask(matte, min_area_ratio=0.05):
    h, w = matte.shape
    binary = (matte > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) 
    mask = (labels == largest_label).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask

def directional_top_suppression(matte, height_ratio=0.25, center_weight=1.0, edge_weight=0.0, center_protect_ratio=0.4):
    h, w = matte.shape
    fade_height = int(h * height_ratio)
    vertical = np.linspace(edge_weight, center_weight, fade_height).reshape(-1, 1)
    x = np.linspace(-1, 1, w)
    horizontal = 1 - np.abs(x)
    horizontal = horizontal * (center_weight - edge_weight) + edge_weight
    fade_mask = vertical @ horizontal[None, :]
    full_mask = np.ones_like(matte)
    full_mask[:fade_height, :] = fade_mask
    protect_width = int(w * center_protect_ratio)
    cx = w // 2
    full_mask[:fade_height, cx - protect_width//2 : cx + protect_width//2] = 1.0

    return matte * full_mask

def remove_bg_modnet(image: Image.Image, modnet=modnet):
    try:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        input_tensor = transform(image).unsqueeze(0).to('cpu')
        with torch.no_grad():
            result = modnet(input_tensor, inference=True)
            if result[2] is None:
                raise ValueError("MODNet returned None for the matte tensor.")
            matte = result[2][0][0].cpu().numpy()
        matte = Image.fromarray((matte * 255).astype(np.uint8)).resize(image.size, Image.LANCZOS)
        matte_np = np.array(matte).astype(np.float32) / 255.0
        matte_np = directional_top_suppression(matte_np, height_ratio=0.25, center_protect_ratio=0.4)
        matte_np = clean_stray_hair_with_morph(matte_np, kernel_size=15, iterations=3)
        matte_np = clean_thin_hairs(matte_np, threshold=0.3, max_thickness=3)
        main_mask = extract_main_head_mask(matte_np)
        matte_np = matte_np * main_mask
        matte_np = soften_matte_edges(matte_np, sigma=3.5)
        foreground = np.array(image).astype(np.float32) / 255.0
        white_bg = np.ones_like(foreground)
        comp = foreground * matte_np[..., None] + white_bg * (1 - matte_np[..., None])
        comp = (comp * 255).astype(np.uint8)

        return Image.fromarray(comp)

    except Exception as e:
        raise


def remove_small_isolated_regions(matte, min_size=500):
    binary = matte > 0.5
    cleaned = morphology.remove_small_objects(binary, min_size=min_size)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_size)
    return cleaned.astype(np.float32) * matte


def clean_thin_hairs(matte, threshold=0.3, max_thickness=3):
    binary = matte > threshold
    skeleton = morphology.skeletonize(binary)
    distance = distance_transform_edt(binary)
    thin_structures = (distance <= max_thickness) & skeleton
    matte_clean = matte.copy()
    matte_clean[thin_structures] = 0
    matte_clean = cv2.GaussianBlur(matte_clean, (5, 5), 0)
    return np.clip(matte_clean, 0, 1)

def clean_alpha_mask(pha):
    pha = np.clip(pha * 1.15, 0, 1)
    pha_binary = (pha > 0.7).astype(np.float32)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((pha_binary * 255).astype(np.uint8), connectivity=8)
    clean_mask = np.zeros_like(pha_binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 1000:
            clean_mask[labels == i] = 1.0
    clean_mask = cv2.GaussianBlur(clean_mask, (7,7), 0)
    clean_mask = np.clip(clean_mask, 0, 1)

    return clean_mask




def apply_studio_editing(image: Image):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.0)  
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.8)
    image = image.filter(ImageFilter.SMOOTH_MORE)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.0)

    return image

def align_face(image, landmarks):
    angle = np.degrees(np.arctan2(0, 0))
    return image.rotate(angle, resample=Image.BICUBIC, expand=True)

def refine_hair_edges(image):
    img_array = np.array(image)
    if img_array.shape[2] != 4:
        raise ValueError("Image must have 4 channels (RGBA)")

    rgb = img_array[:, :, :3]
    alpha = img_array[:, :, 3]
    white_bg = np.full_like(rgb, 255)
    alpha_norm = alpha.astype(np.float32) / 255.0
    comp_rgb = (rgb * alpha_norm[..., None] + white_bg * (1 - alpha_norm[..., None])).astype(np.uint8)
    lab = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]
    blurred = cv2.GaussianBlur(lightness, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 50)
    _, mask_light = cv2.threshold(lightness, 180, 255, cv2.THRESH_BINARY)
    _, mask_dark = cv2.threshold(lightness, 70, 255, cv2.THRESH_BINARY_INV)
    stray_zone = cv2.bitwise_or(mask_light, mask_dark)
    h, w = lightness.shape
    top_mask = np.zeros_like(lightness)
    top_h = int(h * 0.4)
    top_mask[:top_h, :] = 255
    side_w = int(w * 0.15)
    top_mask[:, :side_w] = 255
    top_mask[:, -side_w:] = 255
    stray_edges = cv2.bitwise_and(edges, stray_zone)
    stray_top = cv2.bitwise_and(stray_edges, top_mask)
    dense_hair = cv2.inRange(lightness, 40, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dense_hair = cv2.morphologyEx(dense_hair, cv2.MORPH_CLOSE, kernel, iterations=2)
    stray_top = cv2.bitwise_and(stray_top, cv2.bitwise_not(dense_hair))
    kernel_small = np.ones((2, 2), np.uint8)
    stray_top = cv2.dilate(stray_top, kernel_small, iterations=1)
    stray_top = cv2.morphologyEx(stray_top, cv2.MORPH_OPEN, kernel_small, iterations=1)
    feather_mask = cv2.GaussianBlur(stray_top, (9, 9), 0)
    feather_mask = cv2.normalize(feather_mask, None, 0, 255, cv2.NORM_MINMAX)
    blend_factor = (feather_mask.astype(np.float32) / 255.0)[:, :, None]
    new_alpha = (alpha * (1 - blend_factor.squeeze() * 0.7)).astype(np.uint8)
    result = np.dstack((rgb, new_alpha))
    
    return Image.fromarray(result)

def calculate_face_position(cropped_face, landmarks):
    face_height = cropped_face.height
    face_width = cropped_face.width
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    eye_center_y = (left_eye[1] + right_eye[1]) // 2
    head_height = face_height
    min_head_height_px = int(25.4 * 11.81)
    max_head_height_px = int(35 * 11.81)
    if head_height < min_head_height_px or head_height > max_head_height_px:
        scaling_factor = min_head_height_px / head_height if head_height < min_head_height_px else max_head_height_px / head_height
        new_height = int(face_height * scaling_factor)
        new_width = int(face_width * scaling_factor)
        cropped_face = cropped_face.resize((new_width, new_height), Image.LANCZOS)

    return cropped_face

def feather_edges(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    mask = image.split()[3]
    blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=1))
    feathered_image = Image.new("RGBA", image.size, (255, 255, 255, 0))
    feathered_image.paste(image, (0, 0), blurred_mask)

    return feathered_image


def smooth_edges(image):
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoothed_mask = np.zeros_like(mask)
    cv2.drawContours(smoothed_mask, contours, -1, 255, thickness=cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, kernel)
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel)
    result = cv2.bitwise_and(image_cv, image_cv, mask=smoothed_mask)
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    return result_pil

def remove_shadows_clahe(image):
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    return result_pil

def apply_studio(image):

    img_array = np.array(image.convert("RGB"))
    smooth = cv2.bilateralFilter(img_array, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    return Image.fromarray(sharpened)

def remove_hair_feathers(image):

    img_array = np.array(image)
    if img_array.shape[2] != 4:
        raise ValueError("Image must have 4 channels (RGBA)")

    rgb = img_array[:, :, :3]
    alpha = img_array[:, :, 3]
    white_bg = np.full_like(rgb, 255)
    alpha_norm = alpha.astype(np.float32) / 255.0
    comp_rgb = (rgb * alpha_norm[..., None] + white_bg * (1 - alpha_norm[..., None])).astype(np.uint8)
    gray = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 15, 60)
    mask_light = cv2.inRange(gray, 150, 255)
    mask_dark = cv2.inRange(gray, 60, 150)
    stray_zone = cv2.bitwise_or(mask_light, mask_dark)
    h, w = gray.shape
    top_mask = np.zeros_like(gray)
    top_h = int(h * 0.3)
    top_mask[:top_h, :] = 255
    stray_top = cv2.bitwise_and(edges, stray_zone)
    stray_top = cv2.bitwise_and(stray_top, top_mask)
    dense_hair_mask = cv2.inRange(gray, 50, 130)
    stray_top = cv2.bitwise_and(stray_top, cv2.bitwise_not(dense_hair_mask))
    kernel = np.ones((3, 3), np.uint8)
    stray_top = cv2.dilate(stray_top, kernel, iterations=1)
    stray_top = cv2.morphologyEx(stray_top, cv2.MORPH_CLOSE, kernel, iterations=2)
    feather = cv2.GaussianBlur(stray_top, (11, 11), 0)
    blend_mask = (feather.astype(np.float32) / 255.0)[:, :, None]
    new_alpha = (alpha * (1 - blend_mask.squeeze())).astype(np.uint8)
    result = np.dstack((rgb, new_alpha))

    return Image.fromarray(result)

def smooth_hair_blending(image):
    img_array = np.array(image)
    if img_array.shape[2] != 4:
        raise ValueError("Image must have 4 channels (RGBA)")

    rgb = img_array[:, :, :3]
    alpha = img_array[:, :, 3]
    white_bg = np.full_like(rgb, 255)
    alpha_norm = alpha.astype(np.float32) / 255.0
    comp_rgb = (rgb * alpha_norm[..., None] + white_bg * (1 - alpha_norm[..., None])).astype(np.uint8)
    gray = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 15, 60)
    mask_light = cv2.inRange(gray, 150, 255)
    mask_dark = cv2.inRange(gray, 60, 150)
    stray_zone = cv2.bitwise_or(mask_light, mask_dark)
    h, w = gray.shape
    top_mask = np.zeros_like(gray)
    top_h = int(h * 0.24)
    top_mask[:top_h, :] = 255
    stray_top = cv2.bitwise_and(edges, stray_zone)
    stray_top = cv2.bitwise_and(stray_top, top_mask)
    dense_hair_mask = cv2.inRange(gray, 50, 130)
    stray_top = cv2.bitwise_and(stray_top, cv2.bitwise_not(dense_hair_mask))
    kernel = np.ones((3, 3), np.uint8)
    stray_top = cv2.dilate(stray_top, kernel, iterations=1)
    stray_top = cv2.morphologyEx(stray_top, cv2.MORPH_CLOSE, kernel, iterations=1)
    feather = cv2.GaussianBlur(stray_top, (37, 37), 0)
    blend_mask = (feather.astype(np.float32) / 255.0)[:, :, None]
    blend_strength = 0.6
    blend_mask = np.clip(blend_mask, 0, blend_strength)
    blended_rgb = (rgb * (1 - blend_mask) + 255 * blend_mask).astype(np.uint8)
    new_alpha = (alpha * (1 - blend_mask.squeeze()) + 255 * blend_mask.squeeze() * 0.02).astype(np.uint8)
    result = np.dstack((blended_rgb, new_alpha))
    return Image.fromarray(result)
 

@app.post("/convert-heic-to-jpeg")
async def convert_heic_to_jpeg(file: UploadFile = File(...)):
    try:
        if file.filename.lower().endswith(('.jpg', '.jpeg')):
            image = Image.open(file.file)
            image = ImageOps.exif_transpose(image)
            if(image.width > 1200 and image.height > 1600):
                image = ImageOps.fit(image, (1200,1600), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG", quality=100)
            img_bytes.seek(0)
            
            return Response(
                content=img_bytes.getvalue(),
                media_type="image/jpeg",
                status_code=200
            )
        image = Image.open(file.file)
        image = ImageOps.exif_transpose(image)
        if(image.width > 1200 and image.height > 1600):
            image = ImageOps.fit(image, (1200,1600), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        if image.mode != 'RGB':
            image = image.convert("RGB")
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG", quality=100)
        img_bytes.seek(0)
        return Response(
            content=img_bytes.getvalue(),
            media_type="image/jpeg",
            status_code=200
        )
    
    except Exception as e:
        print(e)
        return {"error": str(e)}

def smooth_skin(pil_img):
    img_np = np.array(pil_img)
    blurred = cv2.bilateralFilter(img_np, 9, 75, 75)
    mask = np.zeros(img_np.shape, dtype=np.float32)
    cv2.illuminationChange(img_np, blurred, mask=mask, alpha=0.2, beta=0.4)
    smoothed = cv2.addWeighted(img_np, 0.8, blurred, 0.2, 0)
    return Image.fromarray(smoothed)

def get_tight_passport_crop(image, landmarks, target_head_ratio=0.63,
                          eye_level_ratio=0.43, final_size=600,
                          min_zoom=1.0, max_zoom=1.5):
    width, height = image.size
    face_x, face_y, face_w, face_h = landmarks['box']
    chin_y = face_y + face_h
    eye_y = (landmarks['keypoints']['left_eye'][1] + landmarks['keypoints']['right_eye'][1]) // 2
    top_of_head_y = face_y - int(face_h * 0.2)
    actual_head_height = chin_y - top_of_head_y
    desired_head_height_pixels = final_size * target_head_ratio
    scale = desired_head_height_pixels / actual_head_height
    shoulder_width = face_w * 1.7
    body_height = chin_y - top_of_head_y + (chin_y - top_of_head_y) * 0.5
    cx = face_x + face_w // 2
    crop_width = shoulder_width
    crop_height = body_height
    width_zoom = final_size / (crop_width * scale)
    height_zoom = final_size / (crop_height * scale)
    required_zoom = max(width_zoom, height_zoom)
    zoom_factor = min(max(min_zoom, required_zoom), max_zoom)
    scaled_crop_width = crop_width * zoom_factor
    scaled_crop_height = crop_height * zoom_factor
    crop_x1 = cx - scaled_crop_width / 2
    crop_x2 = cx + scaled_crop_width / 2
    crop_y1 = eye_y - (eye_level_ratio * scaled_crop_height)
    crop_y2 = crop_y1 + scaled_crop_height
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(width, crop_x2)
    crop_y2 = min(height, crop_y2)
    if crop_x2 - crop_x1 < scaled_crop_width or crop_y2 - crop_y1 < scaled_crop_height:
        actual_width = crop_x2 - crop_x1
        actual_height = crop_y2 - crop_y1
        width_zoom = final_size / (actual_width * scale)
        height_zoom = final_size / (actual_height * scale)
        zoom_factor = min(max(min_zoom, min(width_zoom, height_zoom)), max_zoom)
        scaled_crop_width = crop_width * zoom_factor
        scaled_crop_height = crop_height * zoom_factor
        crop_x1 = max(0, cx - scaled_crop_width / 2)
        crop_x2 = min(width, cx + scaled_crop_width / 2)
        crop_y1 = max(0, eye_y - (eye_level_ratio * scaled_crop_height))
        crop_y2 = min(height, crop_y1 + scaled_crop_height)

    return (int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2))

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_array = np.array(input_image, dtype=np.uint8)
        result = modelFace.get(input_array)
        if len(result) == 0:
            return JSONResponse(content={"error":"No face detected"}, status_code=400)
        elif len(result) > 1:
            return JSONResponse(content={"error": "Multiple faces detected"}, status_code=401)

        cropped_face = get_uk_passport_crop(input_array, result[0])
        # no_bg_image = remove_bg_rvm_cpu(Image.fromarray(cropped_face))
        no_bg_image = remove_bg_modnet(Image.fromarray(cropped_face))
        smoothed_image = feather_edges(no_bg_image)
        no_hair_image = smooth_hair_blending(smoothed_image)
        face_resized = no_hair_image
        white_background = (255, 255, 255)
        passport_background = Image.new("RGB", (1200, 1200), white_background)
        paste_x = (1200 - face_resized.width) // 2
        paste_y = (1200 - face_resized.height) // 2
        face_resized = face_resized.convert("RGBA")
        passport_background.paste(face_resized, (paste_x, paste_y), face_resized.split()[3])
        passport_background=passport_background.resize((1200, 1200), Image.LANCZOS)
        passport_background = apply_studio_editing(passport_background)
        no_bg_image_np = np.array(passport_background)[:, :, ::-1]
        _, _, restored_img = gfpgan_model.enhance(
            no_bg_image_np, 
            has_aligned=False, 
            only_center_face=False, 
            paste_back=True,
            weight=0.3
        )
        original_h, original_w = no_bg_image_np.shape[:2]
        if isinstance(restored_img, list):
            restored_img = restored_img[0]
        if restored_img.shape != no_bg_image_np.shape:
            restored_img = cv2.resize(
                restored_img,
                (original_w, original_h), 
                interpolation=cv2.INTER_LANCZOS4  
            )
        # alpha = 0.5 
        # natural_result = cv2.addWeighted(
        #     no_bg_image_np, alpha,
        #     restored_img, 1 - alpha,
        #     0
        # )
        enhanced_img = Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
        img_byte_arr = io.BytesIO()
        enhanced_img.save(img_byte_arr, format="PNG", quality=100, dpi=(300, 300))
        img_byte_arr.seek(0)

        return Response(content=img_byte_arr.getvalue(), media_type="image/png", status_code=200)

    except Exception as e:
        print(e)
        return Response(content=f"Error: {str(e)}", status_code=500)


@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    try:
        input_image = Image.open(io.BytesIO(await file.read()))
        input_array = np.array(input_image)
        output_array = remove.remove(input_array)
        output_image = Image.fromarray(output_array)
        img_io = io.BytesIO()
        output_image.save(img_io, format="PNG")
        img_io.seek(0)

        return Response(content=img_io.getvalue(), media_type="image/png", status_code=200)
   
    except Exception as e:
        return Response(content=f"Error: {str(e)}", status_code=500)
    

# Pydantic model for creating a payment intent
class PaymentRequest(BaseModel):
    amount: int
    currency: str
    email: str
    phone: str

# Pydantic model for confirming a payment
class ConfirmPaymentRequest(BaseModel):
    payment_intent_id: str  # Payment Intent ID from the client
    payment_method: str  # Payment Method ID from the client

@app.post("/create-payment-intent")
async def create_payment_intent(payment_request: PaymentRequest):
    try:
        # Create a PaymentIntent with metadata
        intent = stripe.PaymentIntent.create(
            amount=payment_request.amount,
            currency=payment_request.currency,
            receipt_email=payment_request.email,  # Send receipt to this email
            metadata={
                "email": payment_request.email,
                "phone": payment_request.phone,
            },
        )
        return {"client_secret": intent["client_secret"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/create-and-send-invoice/")
async def create_and_send_invoice(email: str, name: str = "Unknown User", amount: int = 5000):
    try:
        # Step 1: Check if customer exists
        customers = stripe.Customer.list(email=email)
        if customers.data:
            customer = customers.data[0]  # Use existing customer
        else:
            # Step 2: Create customer if not exists
            customer = stripe.Customer.create(email=email, name=name)

        # Step 3: Create an Invoice Item
        stripe.InvoiceItem.create(
            customer=customer.id,
            amount=amount,  # Amount in cents
            currency="usd",
            description="Photo maker invoice"
        )

        # Step 4: Create an Invoice
        invoice = stripe.Invoice.create(
            customer=customer.id,
            auto_advance=True  # Automatically finalize the invoice
        )

        # Step 5: Finalize & Send the Invoice
        stripe.Invoice.finalize_invoice(invoice.id)
        stripe.Invoice.send_invoice(invoice.id)

        return {
            "message": "Invoice sent successfully",
            "customer_id": customer.id,
            "invoice_id": invoice.id,
            "amount": amount / 100,
            "currency": "USD",
        }

    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")


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


