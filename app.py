"""
Simple AI-powered cutline using rembg (background removal model)
This is the easiest to setup and provides excellent results for most images.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os, uuid
from typing import List, Tuple
from PIL import Image, ImageDraw
import io
import base64

# AI background removal
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
    print("✅ AI-powered segmentation available (rembg)")
except ImportError as e:
    REMBG_AVAILABLE = False
    print(f"⚠️  AI segmentation not available: {e}")
    print("💡 To enable AI features, install: pip install rembg onnxruntime")

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED = {"png", "jpg", "jpeg", "gif"}

def _good(fname: str) -> bool:
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED

def _resize_image(image_path: str, max_dimension: int = 500) -> str:
    """
    Resize image to fit within max_dimension while maintaining aspect ratio.
    Returns the path to the resized image (overwrites original).
    """
    try:
        # Open image with PIL for better format support
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (handles various formats)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get current dimensions
            width, height = img.size
            
            # Check if resizing is needed
            if max(width, height) <= max_dimension:
                print(f"Image {width}x{height} is already small enough, no resize needed")
                return image_path
            
            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = max_dimension
                new_height = int((height * max_dimension) / width)
            else:
                new_height = max_dimension
                new_width = int((width * max_dimension) / height)
            
            print(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
            
            # Resize with high quality
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save back to the same path
            resized_img.save(image_path, 'JPEG', quality=90, optimize=True)
            
            return image_path
            
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image_path  # Return original path if resize fails

def _smooth_spline(points: List[Tuple[int, int]], factor=0.35) -> List[Tuple[int, int]]:
    """Apply spline smoothing to points"""
    if len(points) < 4:
        return points
    try:
        from scipy import interpolate
    except ImportError:
        return points

    arr = np.asarray(points, dtype=np.float32)
    cx, cy = arr.mean(axis=0)
    arr_centered = arr - (cx, cy)

    # Create parameter based on cumulative chord length
    diffs = np.diff(arr, axis=0, append=arr[0:1])
    chord_lengths = np.linalg.norm(diffs, axis=1)
    t = np.concatenate(([0], np.cumsum(chord_lengths)))[:-1]
    t = t / (t[-1] + 1e-6)

    # Use periodic spline
    smoothing_value = len(points) * factor
    tck, _ = interpolate.splprep([arr_centered[:, 0], arr_centered[:, 1]], 
                                s=smoothing_value, per=True, u=t)
    
    # Generate smooth points
    num_points = max(300, len(points) * 4)
    u_fine = np.linspace(0, 1, num_points, endpoint=False)
    x_smooth, y_smooth = interpolate.splev(u_fine, tck)
    
    result = [(int(x + cx), int(y + cy)) for x, y in zip(x_smooth, y_smooth)]
    return result

def _ai_segment_rembg(image_path: str, model_name='u2net') -> np.ndarray:
    """Use rembg to remove background and get clean mask"""
    
    # Available models:
    # 'u2net' - General use (default)
    # 'u2netp' - Lighter version
    # 'u2net_human_seg' - Optimized for people
    # 'silueta' - Good for objects
    # 'isnet-general-use' - Latest, very good
    
    with open(image_path, 'rb') as f:
        input_data = f.read()
    
    # Create session with specific model
    session = new_session(model_name)
    
    # Remove background
    output_data = remove(input_data, session=session)
    
    # Convert to numpy array
    img_array = np.frombuffer(output_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    
    # Extract alpha channel as mask
    if img.shape[2] == 4:
        mask = img[:, :, 3]
    else:
        # Fallback: convert to grayscale and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    return mask

def _contour_from_mask(mask: np.ndarray, pad: int = 12) -> List[Tuple[int, int]]:
    """Extract smooth contour from AI-generated mask"""
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Add padding (bleed)
    if pad > 0:
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad*2+1, pad*2+1))
        mask = cv2.dilate(mask, dilation_kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        h, w = mask.shape[:2]
        return [(10, 10), (w-10, 10), (w-10, h-10), (10, h-10)]
    
    # Get largest contour
    cnt = max(contours, key=cv2.contourArea)
    
    # Simplify contour
    peri = cv2.arcLength(cnt, True)
    epsilon = 0.002 * peri  # Very fine detail
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # Convert to point list
    pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
    
    # Apply heavy smoothing for professional results
    if len(pts) > 8:
        pts = _smooth_spline(pts, 1.0)  # Maximum smoothing
    
    return pts

def _fallback_contour_points(path: str, size: str = "medium") -> List[Tuple[int, int]]:
    """Your original fallback method"""
    PAD = {"small": 32, "medium": 40, "large": 50}  # More generous padding: small=old large, medium=more, large=most
    pad = PAD.get(size, PAD["medium"])

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return []

    # Background detection
    bg = np.median(np.vstack([
        img[:5, :, :].reshape(-1, 3),
        img[-5:, :, :].reshape(-1, 3),
        img[:, :5, :].reshape(-1, 3),
        img[:, -5:, :].reshape(-1, 3),
    ]), axis=0)

    diff = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
    mask = (diff > 16).astype(np.uint8) * 255
    
    return _contour_from_mask(mask, pad)

def _contour_points(path: str, size: str = "medium", model: str = "u2net") -> List[Tuple[int, int]]:
    """Main function - uses AI if available"""
    PAD = {"small": 32, "medium": 40, "large": 50}  # More generous padding: small=old large, medium=more, large=most
    pad = PAD.get(size, PAD["medium"])
    
    if REMBG_AVAILABLE:
        try:
            print(f"Using AI model: {model}")
            mask = _ai_segment_rembg(path, model)
            return _contour_from_mask(mask, pad)
        except Exception as e:
            print(f"AI segmentation failed: {e}")
            return _fallback_contour_points(path, size)
    else:
        print("Using fallback method")
        return _fallback_contour_points(path, size)

@app.route("/")
def home():
    status = "AI-powered (rembg)" if REMBG_AVAILABLE else "Fallback method"
    models = ["u2net", "u2netp", "u2net_human_seg", "silueta", "isnet-general-use"] if REMBG_AVAILABLE else []
    
    return jsonify({
        "message": "Cutline API v6 - AI Background Removal + All Shapes",
        "status": "running",
        "segmentation": status,
        "available_models": models,
        "available_shapes": ["contour", "circle", "rectangle", "rounded"],
        "endpoints": {
            "cutline": "/cutline (POST) - Generate shape data",
            "preview": "/preview (POST) - Generate preview image"
        },
        "parameters": {
            "cutline": {
                "file": "image file",
                "shape": "contour/circle/rectangle/rounded",
                "size": "small/medium/large (bleed size, contour only)",
                "model": "AI model name (contour only, if available)",
                "resize": "true/false (optional, resize for performance vs quality)"
            },
            "preview": {
                "file": "image file", 
                "shape": "contour/circle/rectangle/rounded",
                "size": "small/medium/large (bleed size, contour only)",
                "model": "AI model name (contour only, if available)",
                "background": "transparent/white/black",
                "resize": "true/false (optional, resize for performance vs quality)"
            }
        },
        "notes": {
            "contour": "AI-powered perfect cutlines (recommended)",
            "circle": "Perfect circle with drag/scale controls",
            "rectangle": "Simple rectangle",
            "rounded": "Rectangle with rounded corners"
        }
    })

@app.route("/cutline", methods=["POST"])
def cutline():
    try:
        if "file" not in request.files:
            return jsonify(error="file field missing"), 400
        f = request.files["file"]
        if not f or f.filename == "":
            return jsonify(error="no file selected"), 400
        if not _good(f.filename):
            return jsonify(error="file type not allowed"), 400

        size = request.form.get("size", "medium")
        shape = request.form.get("shape", "contour")  # Get shape selection
        model = request.form.get("model", "u2net")  # Use original working model by default
        
        name = f"{uuid.uuid4()}_{f.filename}"
        path = os.path.join(UPLOAD_DIR, name)
        f.save(path)

        # Optional resize for performance (disabled by default to preserve AI quality)
        resize = request.form.get("resize", "false").lower() == "true"
        if resize:
            path = _resize_image(path, max_dimension=500)

        img = cv2.imread(path)
        if img is None:
            return jsonify(error="invalid image"), 400
        h, w = img.shape[:2]

        # Handle different shapes
        if shape == "contour":
            pts = _contour_points(path, size, model)
            method = f"rembg-{model}" if REMBG_AVAILABLE else "fallback"
            return jsonify(
                type="contour", 
                points=pts, 
                width=w, 
                height=h, 
                size=size,
                method=method,
                model=model if REMBG_AVAILABLE else None,
                point_count=len(pts),
                ai_powered=REMBG_AVAILABLE
            )
        
        elif shape == "circle":
            r = min(w, h) // 2
            return jsonify(type="circle", cx=w // 2, cy=h // 2, r=r, width=w, height=h)
        
        elif shape == "rectangle":
            # Make rectangle smaller than full image (80% of dimensions, centered)
            margin = min(w, h) * 0.1  # 10% margin on each side
            rect_w = w - (2 * margin)
            rect_h = h - (2 * margin)
            rect_x = margin
            rect_y = margin
            return jsonify(type="rectangle", 
                         points=[(rect_x, rect_y), (rect_x + rect_w, rect_y), 
                                (rect_x + rect_w, rect_y + rect_h), (rect_x, rect_y + rect_h)], 
                         width=w, height=h)
        
        elif shape == "rounded":
            # Make rounded rectangle smaller than full image (75% of dimensions, centered)
            margin = min(w, h) * 0.125  # 12.5% margin on each side  
            rect_w = w - (2 * margin)
            rect_h = h - (2 * margin)
            rect_x = margin
            rect_y = margin
            rad = min(rect_w, rect_h) // 4  # 25% corner radius (more rounded)
            return jsonify(type="rounded", 
                         points=[(rect_x, rect_y), (rect_x + rect_w, rect_y), 
                                (rect_x + rect_w, rect_y + rect_h), (rect_x, rect_y + rect_h)], 
                         radius=rad, width=w, height=h)
        
        else:
            return jsonify(error="unknown shape"), 400

    except Exception as e:
        print("ERR /cutline:", repr(e))
        return jsonify(error="internal server error"), 500

@app.route("/preview", methods=["POST"])
def preview():
    """Generate a preview of the final cutline result"""
    try:
        if "file" not in request.files:
            return jsonify(error="file field missing"), 400
        f = request.files["file"]
        if not f or f.filename == "":
            return jsonify(error="no file selected"), 400
        if not _good(f.filename):
            return jsonify(error="file type not allowed"), 400

        size = request.form.get("size", "medium")
        shape = request.form.get("shape", "contour")
        model = request.form.get("model", "u2net")
        background = request.form.get("background", "transparent")  # transparent, white, black, etc.
        
        name = f"{uuid.uuid4()}_{f.filename}"
        path = os.path.join(UPLOAD_DIR, name)
        f.save(path)

        # Optional resize for performance (disabled by default to preserve AI quality)
        resize = request.form.get("resize", "false").lower() == "true"
        if resize:
            path = _resize_image(path, max_dimension=500)

        # Load image with PIL for better processing
        img_pil = Image.open(path).convert("RGBA")
        w, h = img_pil.size

        # Create mask based on shape type
        mask = Image.new("L", (w, h), 0)  # Black mask
        draw = ImageDraw.Draw(mask)

        if shape == "contour":
            pts = _contour_points(path, size, model)
            if len(pts) > 2:
                # Convert points to PIL format
                pil_points = [(int(p[0]), int(p[1])) for p in pts]
                draw.polygon(pil_points, fill=255)
        
        elif shape == "circle":
            r = min(w, h) // 2
            cx, cy = w // 2, h // 2
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=255)
        
        elif shape == "rectangle":
            margin = min(w, h) * 0.1
            rect_w = w - (2 * margin)
            rect_h = h - (2 * margin)
            rect_x = margin
            rect_y = margin
            draw.rectangle([rect_x, rect_y, rect_x + rect_w, rect_y + rect_h], fill=255)
        
        elif shape == "rounded":
            margin = min(w, h) * 0.125
            rect_w = w - (2 * margin)
            rect_h = h - (2 * margin)
            rect_x = margin
            rect_y = margin
            rad = min(rect_w, rect_h) // 4
            draw.rounded_rectangle([rect_x, rect_y, rect_x + rect_w, rect_y + rect_h], 
                                 radius=rad, fill=255)

        # Create background - support multiple material types
        if background == "transparent" or background == "none":
            background_color = (0, 0, 0, 0)
        elif background == "white" or background == "vinyl" or background == "hi-tack-vinyl":
            background_color = (255, 255, 255, 255)
        elif background == "black":
            background_color = (0, 0, 0, 255)
        elif background == "holographic":
            # Use a rainbow-like color for holographic
            background_color = (255, 0, 128, 255)  # Bright pink for holographic effect
        elif background == "glitter":
            background_color = (255, 215, 0, 255)  # Gold
        elif background == "mirror" or background == "brushed-aluminum":
            background_color = (192, 192, 192, 255)  # Silver
        elif background == "pixie-dust":
            background_color = (230, 230, 250, 255)  # Lavender
        elif background == "prismatic":
            background_color = (75, 183, 209, 255)  # Blue
        elif background == "kraft-paper":
            background_color = (210, 180, 140, 255)  # Tan
        elif background == "low-tack-vinyl":
            background_color = (240, 240, 240, 255)  # Light gray
        elif background == "reflective":
            background_color = (224, 224, 224, 255)  # Light silver
        elif background == "glow-in-dark":
            background_color = (191, 255, 0, 255)  # Bright green
        else:
            # Default to transparent for unknown materials
            background_color = (0, 0, 0, 0)

        # Convert mask to numpy array for processing
        mask_array = np.array(mask)
        
        # Start with transparent canvas
        result = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        
        # EXTENSIVE DEBUG: Let's see what's really happening
        mask_white_pixels = np.sum(mask_array > 0)
        mask_total_pixels = mask_array.size
        print(f"🔍 DEBUG: Shape={shape}, Mask size={mask_array.shape}")
        print(f"🔍 DEBUG: Mask has {mask_white_pixels} white pixels out of {mask_total_pixels} total ({mask_white_pixels/mask_total_pixels*100:.1f}%)")
        print(f"🔍 DEBUG: Background color={background_color}")
        
        # Save debug mask to see what it looks like
        debug_mask_path = os.path.join(UPLOAD_DIR, f"debug_mask_{shape}.png")
        Image.fromarray(mask_array, "L").save(debug_mask_path)
        print(f"🔍 DEBUG: Saved mask to {debug_mask_path}")
        
        # THE ISSUE: For contour, the mask is probably correct but we need to make sure
        # Let's check if this is really a contour issue by testing both ways
        if shape == "contour":
            # Check if mask seems inverted by looking at corners
            corner_pixels = [
                mask_array[0, 0], mask_array[0, -1], 
                mask_array[-1, 0], mask_array[-1, -1]
            ]
            corner_white_count = sum(1 for p in corner_pixels if p > 0)
            print(f"🔍 DEBUG: Corner pixels white count: {corner_white_count}/4")
            
            # If most corners are white, the mask is probably inverted
            if corner_white_count >= 3:
                print("🔍 DEBUG: INVERTING contour mask (corners are white)")
                mask_array = 255 - mask_array
        
        # SIMPLE TEST: Create result step by step with debugging
        print(f"🔍 DEBUG: Creating result image...")
        
        # Step 1: Just the background where mask is white
        if background != "transparent":
            print(f"🔍 DEBUG: Adding background material...")
            
            # Create background layer
            bg_layer = np.zeros((h, w, 4), dtype=np.uint8)
            mask_positions = mask_array == 255  # Only where mask is exactly 255 (white)
            
            # Handle gradient materials (holographic, prismatic)
            if background == "holographic":
                # Create holographic gradient effect
                for y in range(h):
                    for x in range(w):
                        if mask_positions[y, x]:
                            # Create rainbow gradient based on position
                            progress = (x + y) / (w + h)  # Diagonal gradient
                            if progress < 0.33:
                                # Red to Green
                                t = progress / 0.33
                                r = int(255 * (1 - t))
                                g = int(255 * t)
                                b = 0
                            elif progress < 0.66:
                                # Green to Blue
                                t = (progress - 0.33) / 0.33
                                r = 0
                                g = int(255 * (1 - t))
                                b = int(255 * t)
                            else:
                                # Blue to Purple
                                t = (progress - 0.66) / 0.34
                                r = int(128 * t)
                                g = 0
                                b = 255
                            bg_layer[y, x] = [r, g, b, 255]
            elif background == "prismatic":
                # Create prismatic gradient effect
                for y in range(h):
                    for x in range(w):
                        if mask_positions[y, x]:
                            # Create diagonal rainbow gradient
                            progress = (x / w + y / h) / 2
                            if progress < 0.5:
                                # Red to Cyan
                                t = progress / 0.5
                                r = int(255 * (1 - t) + 75 * t)
                                g = int(183 * t)
                                b = int(209 * t)
                            else:
                                # Cyan to Blue
                                t = (progress - 0.5) / 0.5
                                r = int(75 * (1 - t) + 69 * t)
                                g = int(183 * (1 - t) + 123 * t)
                                b = int(209 * (1 - t) + 177 * t)
                            bg_layer[y, x] = [r, g, b, 255]
            else:
                # Solid color materials
                bg_layer[mask_positions] = background_color
            
            print(f"🔍 DEBUG: Background applied to {np.sum(mask_positions)} pixels")
            bg_img = Image.fromarray(bg_layer, "RGBA")
            result = Image.alpha_composite(result, bg_img)
        
        # Step 2: Original image only where mask is white
        print(f"🔍 DEBUG: Adding original image...")
        img_layer = np.array(img_pil, dtype=np.uint8)
        
        # Make image transparent where mask is black (0)
        mask_positions = mask_array == 255
        img_layer[:, :, 3] = np.where(mask_positions, img_layer[:, :, 3], 0)
        
        print(f"🔍 DEBUG: Image applied to {np.sum(mask_positions)} pixels")
        img_masked = Image.fromarray(img_layer, "RGBA")
        result = Image.alpha_composite(result, img_masked)

        # Convert to base64 for response
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Clean up
        os.remove(path)

        return jsonify(
            success=True,
            preview_image=f"data:image/png;base64,{img_base64}",
            shape=shape,
            background=background,
            dimensions={"width": w, "height": h}
        )

    except Exception as e:
        print("ERR /preview:", repr(e))
        return jsonify(error="internal server error"), 500

if __name__ == "__main__":
    print(f"AI Status: {'Available' if REMBG_AVAILABLE else 'Not available'}")
    if REMBG_AVAILABLE:
        print("Available models: u2net, u2netp, u2net_human_seg, silueta, isnet-general-use")
    
    # Get port from environment variable for deployment
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(host='0.0.0.0', port=port, debug=debug)
