import halcon as ha
import numpy as np

print("Attributes in 'halcon' module:")
# print(dir(ha)) # output might be huge, let's filter or print carefully

# Check for HImage-like names
attrs = dir(ha)
h_images = [a for a in attrs if 'HImage' in a or 'Image' in a]
print("Image related attributes:", h_images)

# Check for common operators
print("Has gen_image1?", 'gen_image1' in attrs)
print("Has GenImage1?", 'GenImage1' in attrs)

