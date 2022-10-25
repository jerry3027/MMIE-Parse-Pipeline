# from decimer_segmentation import segment_chemical_structures, segment_chemical_structures_from_file
import os
import cv2
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'

path = './Outputs/PSC/acs.macromol.5b00098/figures/'

images = os.listdir(path)

images.sort()

image_1 = os.path.join(path, images[0])
# cv_image = cv2.imread(image_1)
# segments = segment_chemical_structures_from_file(image_1, expand=True)

print(pytesseract.image_to_string(image_1))
