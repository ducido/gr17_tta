import cv2
import imageio
import numpy as np


H = 400
W = 400

img = np.zeros((H, W, 3), dtype=np.uint8)

# top-left: RED
img[:200, :200] = [0, 0, 255]

# top-right: GREEN
img[:200, 200:] = [0, 255, 0]

# bottom-left: BLUE
img[200:, :200] = [255, 0, 0]

# bottom-right: WHITE
img[200:, 200:] = [255, 255, 255]

cv2.putText(
    img,
    "BGR",
    (130, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 0),
    2,
)

print("top-left pixel:", img[100, 100])
print("top-right pixel:", img[100, 300])
print("bottom-left pixel:", img[300, 100])

# --------------------------------------------------
# save by cv2
# --------------------------------------------------

cv2.imwrite("cv2_saved.png", img)

# --------------------------------------------------
# save by imageio directly
# --------------------------------------------------

imageio.imwrite("imageio_saved.png", img)

# --------------------------------------------------
# save by imageio after BGR->RGB conversion
# --------------------------------------------------

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

imageio.imwrite("imageio_saved_rgb.png", img_rgb)

# --------------------------------------------------
# save video without conversion
# --------------------------------------------------

writer = imageio.get_writer(
    "video_bgr.mp4",
    fps=1,
    codec="libx264",
)

for _ in range(10):
    writer.append_data(img)

writer.close()

# --------------------------------------------------
# save video with conversion
# --------------------------------------------------

writer = imageio.get_writer(
    "video_rgb.mp4",
    fps=1,
    codec="libx264",
)

for _ in range(10):
    writer.append_data(img_rgb)

writer.close()

print("Done")