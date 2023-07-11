import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from run import model_for_pruning

def load_image(img_path, show=False):

    img = Image.open(img_path)
    # img = img.resize((250, 200))
    img = img.convert('RGB') 
    img_array = np.array(img, dtype=np.float32)
    img_array = np.resize(img_array, (250, 200, 3))
    print(img_array.shape)              # (height, width, channels)
    img_tensor = np.expand_dims(img_array, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.0                                    

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor
  
img_path = 'C:/Users/danil/Documents/imgintegrador/valid/fitabranca/'
arquivos = os.listdir(img_path)
primeiro_arquivo = arquivos[0]
img_path = os.path.join(img_path, primeiro_arquivo)

new_image = load_image(img_path)

pred = model_for_pruning.predict(new_image)

print(pred)

#output1 
# [[0.65795904 0.342041  ]]


