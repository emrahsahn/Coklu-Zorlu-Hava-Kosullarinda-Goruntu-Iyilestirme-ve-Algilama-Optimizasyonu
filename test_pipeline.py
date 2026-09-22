import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Create a dummy image (256x256 BGR)
img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
# Add some "rain" like streaks (white lines)
cv2.line(img, (50, 0), (100, 256), (255, 255, 255), 2)
cv2.line(img, (200, 0), (150, 256), (255, 255, 255), 2)

cv2.imwrite("dummy_rain.jpg", img)

from src_rain_and_snow.pipeline import RainSnowPipeline

print("Initializing Pipeline...")
pipeline = RainSnowPipeline()
print("Running Pipeline...")
pipeline.process("dummy_rain.jpg")
print("Verification complete.")
