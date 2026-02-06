""" utils.py fungerar som projektets matematiska motor och optiker. 
Den sköter två huvuduppgifter: Spline-hantering, som förvandlar enkla koordinater till mjuka, naturliga kurvor (likt riktiga vägar), 
och Bildbehandling, som förbereder rådata från simulatorns kamera för AI-hjärnan. 
Genom att beskära bort oviktig information (som horisonten) och normalisera pixelvärden skapas en ren och effektiv indata som gör att 
modellen kan fokusera helt på att tolka linjens kurvatur."""

import numpy as np
import cv2
from scipy.interpolate import CubicSpline

def generate_random_spline(length=10.0, num_points=5, complexity=2.0):
    """
    Skapar en helt slumpmässig bana för bilen att följa under träning.
    Använder CubicSpline för att garantera mjuka svängar.
    """
    # Skapa x-punkter jämnt fördelade längs banans längd
    x = np.linspace(0, length, num_points)
    # Slumpa y-värden (svängar åt höger/vänster) baserat på komplexitet
    y = np.random.uniform(-complexity, complexity, size=num_points)
    
    # Tvingar de första två punkterna att vara raka (y=0) 
    # så att bilen alltid får en stabil startposition.
    y[0] = 0
    y[1] = 0
    
    # Skapar själva funktionen för kurvan
    cs = CubicSpline(x, y, bc_type='clamped')
    return cs, x

def create_custom_spline(points):
    """
    Skapar en bana utifrån en lista med egna [x, y] koordinater.
    Används främst av sandbox.py när du ritar banan själv.
    """
    pts = np.array(points)
    x = pts[:, 0]
    y = pts[:, 1]
    # 'clamped' gör att kurvans start och slut blir mer kontrollerade
    cs = CubicSpline(x, y, bc_type='clamped')
    return cs, x

def preprocess_image(image, target_size=(160, 120), grayscale=False):
    """
    Omvandlar råbilder från simulatorn till ett format AI:n kan förstå.
    Innehåller beskärning, storleksändring och normalisering.
    """
    # Säkerställ att bilden är i rätt format för OpenCV
    image = np.asarray(image, dtype=np.uint8)
    
    # Om simulatorn ger RGBA (4 kanaler), konvertera till vanlig RGB (3 kanaler)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Valfri konvertering till gråskala (används om modellen ska vara färgblind)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
    
    # 1. BESKÄRNING (CROP): Ta bort den översta tredjedelen av bilden.
    # AI:n behöver bara se marken och linjen, inte väggar eller horisont.
    h, w = image.shape[:2]
    crop = image[int(h/3):, :]
    
    # 2. RESIZE: Ändra upplösningen till mål-storleken (t.ex. 160x120)
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    
    # 3. NORMALISERING: Ändra pixelvärden från 0-255 till 0.0-1.0.
    # Detta är kritiskt för att det neurala nätverket ska kunna lära sig effektivt.
    normalized = resized.astype(np.float32) / 255.0
    
    # 4. TRANSPOSE: Ändra ordning på dimensionerna till (Kanaler, Höjd, Bredd).
    # PyTorch/Stable Baselines 3 kräver kanalen först, medan OpenCV har den sist.
    if len(normalized.shape) == 2: # Om gråskala
        return normalized.reshape(1, target_size[1], target_size[0])
    else: # Om RGB
        return normalized.transpose(2, 0, 1).copy()

def add_noise(image, intensity=0.01):
    """
    Lägger till slumpmässigt brus (snö) i bilden.
    Används för 'domain randomization' så att AI:n tål sämre kameror i verkligheten.
    """
    noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 1) # Håll värdena mellan 0 och 1
    return noisy_image

def get_line_point(cs, x_range, t):
    """
    Hämtar ut [x, y] koordinater för en specifik punkt på banan.
    t är ett värde mellan 0 (start) och 1 (mål).
    """
    x_val = t * x_range[-1]
    y_val = cs(x_val)
    return np.array([x_val, y_val])