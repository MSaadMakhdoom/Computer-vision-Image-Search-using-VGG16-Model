from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

# See https://keras.io/api/applications/ for details

class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
    

        # feature (np.ndarray): deep feature with the shape=(4096, )
       
        img = img.resize((224, 224))  #224x224 img as an input
        img = img.convert('RGB')  
        x = image.img_to_array(img)  #  np.array. Height x Width x Channel. dtype=float32
        print("upload image shape ",x.shape)
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C)
        print("-------- upload image  ",x)
        print("upload image shape ",x.shape)

        x = preprocess_input(x)  # Subtracting avg values for each pixel

        print("--> upload image preprocess shape ",x.shape)
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        print("Model predict feature shape ",feature.shape)
        return feature / np.linalg.norm(feature)  # Normalize

