import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Load the parrot.jpg image
image_path = "panda.jpg"
image = imread(image_path)

# Convert image to grayscale
gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

# Compute the Singular Value Decomposition (SVD) of the image
u, s, vh = np.linalg.svd(gray_image, full_matrices=False)

# Save the SVD components as separate images
plt.imsave("u.jpg", u, cmap="gray")
plt.imsave("s.jpg", np.diag(s), cmap="gray")
plt.imsave("vh.jpg", vh, cmap="gray")