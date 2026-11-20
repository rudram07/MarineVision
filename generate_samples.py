import os
import cv2
import numpy as np

def generate_sample_sonar_images():
    output_dir = os.path.join(os.path.dirname(__file__), 'static', 'images', 'samples')
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Sample 1: Sonar Seabed with Metallic Debris and Acoustic Shadow
    # -------------------------------------------------------------
    h, w = 600, 800
    np.random.seed(42)
    
    # 1. Base acoustic seabed texture (sand ripples + speckle)
    x = np.linspace(0, 10 * np.pi, w)
    y = np.linspace(0, 8 * np.pi, h)
    xx, yy = np.meshgrid(x, y)
    ripples = np.sin(xx + 0.5 * np.cos(yy)) * 25 + 110
    
    # Acoustic speckle (Rayleigh-like)
    speckle = np.random.rayleigh(scale=18, size=(h, w))
    sonar_img = ripples + speckle

    # Center nadir (water column: darker acoustic deadzone down the center)
    nadir_width = 40
    nadir_center = w // 2
    for col in range(nadir_center - nadir_width, nadir_center + nadir_width):
        dist = abs(col - nadir_center)
        dim_factor = (dist / nadir_width) ** 0.8
        sonar_img[:, col] = sonar_img[:, col] * dim_factor * 0.4 + 15

    # Target 1: Sunken Metal Frame / Container Debris (left channel)
    # Bright highlight (specular acoustic reflection) + dark acoustic shadow
    obj_x, obj_y, obj_w, obj_h = 220, 240, 70, 45
    sonar_img[obj_y:obj_y+obj_h, obj_x:obj_x+obj_w] += 120  # Bright metallic highlight
    # Acoustic shadow cast away from nadir (towards the left)
    shadow_w = 90
    sonar_img[obj_y+2:obj_y+obj_h-2, max(0, obj_x-shadow_w):obj_x] = 12

    # Target 2: Cylindrical Drum / Pipe Debris (right channel)
    d_x, d_y = 560, 380
    sonar_img[d_y:d_y+30, d_x:d_x+40] += 110
    # Shadow cast towards the right
    sonar_img[d_y+2:d_y+28, d_x+40:d_x+100] = 14

    # Clip to valid uint8 range
    sonar_img = np.clip(sonar_img, 0, 255).astype(np.uint8)

    sample1_path = os.path.join(output_dir, 'sample_sonar_debris.png')
    cv2.imwrite(sample1_path, sonar_img)

    # -------------------------------------------------------------
    # Sample 2: Sonar Seabed with Rocky Outcrop & Acoustic Anomaly
    # -------------------------------------------------------------
    np.random.seed(101)
    ripples2 = np.cos(xx * 0.8 + 0.3 * np.sin(yy)) * 30 + 105
    speckle2 = np.random.rayleigh(scale=22, size=(h, w))
    sonar2 = ripples2 + speckle2

    # Center nadir
    for col in range(nadir_center - 35, nadir_center + 35):
        dist = abs(col - nadir_center)
        dim_factor = (dist / 35) ** 0.8
        sonar2[:, col] = sonar2[:, col] * dim_factor * 0.35 + 18

    # Unusual seabed anomaly crater / scour depression
    cv2.circle(sonar2, (280, 360), 55, 30, -1) # dark scour depression
    cv2.circle(sonar2, (275, 355), 45, 175, 4) # ridge rim

    # Debris boulder cluster with sharp shadow
    sonar2[180:210, 600:640] += 130
    sonar2[182:208, 640:710] = 10

    sonar2 = np.clip(sonar2, 0, 255).astype(np.uint8)
    sample2_path = os.path.join(output_dir, 'sample_sonar_anomaly.png')
    cv2.imwrite(sample2_path, sonar2)

    print(f"Generated sample 1: {sample1_path}")
    print(f"Generated sample 2: {sample2_path}")

if __name__ == '__main__':
    generate_sample_sonar_images()
