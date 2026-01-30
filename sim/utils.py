import numpy as np
import cv2
from scipy.interpolate import CubicSpline

def generate_random_spline(length=10.0, num_points=5, complexity=2.0):
    """
    Generates a random spline for the car to follow.
    Returns a function that takes t in [0, 1] and returns [x, y].
    """
    # Create random waypoints
    x = np.linspace(0, length, num_points)
    y = np.random.uniform(-complexity, complexity, size=num_points)
    
    # Ensure start and end are straight-ish if needed
    y[0] = 0
    y[1] = 0
    
    cs = CubicSpline(x, y, bc_type='clamped')
    return cs, x

def create_custom_spline(points):
    """
    Creates a spline from a list of [x, y] points.
    Points must be sorted by x.
    """
    pts = np.array(points)
    x = pts[:, 0]
    y = pts[:, 1]
    cs = CubicSpline(x, y, bc_type='clamped')
    return cs, x

def preprocess_image(image, target_size=(160, 120), grayscale=False):
    """
    Preprocesses the camera image: crop, optional grayscale, resize, normalize.
    """
    # Ensure image is uint8 for OpenCV
    image = np.asarray(image, dtype=np.uint8)
    
    # Assuming image is from PyBullet (RGB or RGBA)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # 1. Grayscale (Optional now)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
    
    # 2. Crop (bottom 2/3)
    h, w = image.shape[:2]
    crop = image[int(h/3):, :]
    
    # 3. Resize
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    
    # 4. Normalize
    normalized = resized.astype(np.float32) / 255.0
    
    # Transpose to (Channels, Height, Width) for PyTorch/SB3
    if len(normalized.shape) == 2: # Grayscale case
        return normalized.reshape(1, target_size[1], target_size[0])
    else: # RGB case
        return normalized.transpose(2, 0, 1).copy()

def add_noise(image, intensity=0.01):
    """
    Adds Gaussian noise to the image for domain randomization.
    """
    noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

def get_line_point(cs, x_range, t):
    """
    Gets the [x, y] coordinates of the line at progress t (0 to 1).
    """
    x_val = t * x_range[-1]
    y_val = cs(x_val)
    return np.array([x_val, y_val])
