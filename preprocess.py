from tensorflow.keras.preprocessing import image
import numpy as np

def prepare_image(filepath):
    img = image.load_img(filepath, target_size=(128, 128))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img