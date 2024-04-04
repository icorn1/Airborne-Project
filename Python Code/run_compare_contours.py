from contours import load_contour
from test1 import ContourComparer
from main_communication import get_genie_image
import cv2
import warnings

# Filter out the PendingDeprecationWarning
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

comparer = ContourComparer()
model_contour = load_contour("contours/14_mesh_contour.txt")

img = get_genie_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
_, mask = cv2.threshold(gray, 140, 255, 0)
found_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = [contour for contour in found_contours if cv2.contourArea(contour) > 20000]

comparer.set_model_contours([model_contour])
ret = comparer.match_contour_to_model(contours, 30, 1.53, 1, img)
