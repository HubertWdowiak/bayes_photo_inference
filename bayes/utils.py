from CyfroVet.ML_Veterinray_Metrics_Func import Face_detector

from tensorflow.keras.preprocessing import image
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import os
import cv2


def simplify_image(image: np.ndarray, slice_shape: tuple[int, int], pooling_func: callable = np.max):
    sliced_image = slice_img(image, slice_shape)
    if len(image.shape) == 2:
        arr = np.array([pooling_func(sl) for sl in sliced_image]).reshape((image.shape[0] // slice_shape[0],
                                                                           image.shape[1] // slice_shape[1]))
    else:
        shape_0 = shape_1 = int(np.sqrt(image.shape[0]))
        arr = np.array([pooling_func(sl) for sl in sliced_image]).reshape((shape_0 // slice_shape[0],
                                                                           shape_1 // slice_shape[1]))
    return arr


def image_preprocess(filename: str, img_size: tuple[int, int]):
    img = cv2.imread(filename)
    x, x1, x2, y1, y2 = Face_detector(img)
    x = img[y1:y2, x1:x2]
    rgb_image = cv2.resize(x, img_size)
    image = np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])
    standardized_image = (image - np.mean(image)) / np.std(image)
    return standardized_image


def training_preprocessor(input_path: str, res: tuple = (100, 100)):
    img = image.load_img(input_path, target_size=res)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    img /= 255
    return img


def get_heatmap(img_path: str, res=(100, 100)):
    model = tf.keras.models.load_model("/Users/hwdowiak/Desktop/inzynierka/data/my_model4")
    img = training_preprocessor(img_path)
    conv_layer = model.get_layer(index=0)
    heatmap_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])
    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(img)
        argmax = tf.argmax(predictions[0])
        loss = predictions[:, argmax]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    heatmap *= 255
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    heatmap = heatmap.squeeze()
    heatmap = cv2.resize(heatmap, res)
    return heatmap


def get_age_category(age: int) -> str:
    if age < 4:
        return 'junior'
    elif age < 7:
        return 'adult'
    else:
        return 'senior'


def read_data(path: str) -> list[(str, str, str)]:
    disease_dirs = os.listdir(path)
    disease_dirs.remove('.DS_Store')
    final_paths = []
    for disease_dir in disease_dirs:
        current_diseases = []
        for disease in disease_dir.split(';'):
            if disease != '.DS_Store':
                current_diseases.append(disease)
        for year_dir in os.listdir(f'{path}/{disease_dir}'):
            if year_dir != '.DS_Store':
                final_paths.append((f'{path}/{disease_dir}/{year_dir}', current_diseases,
                                    get_age_category(int(year_dir[:-1]))))

    return final_paths

def slice_img(image: np.ndarray, kernel_size: tuple):
    tile_height, tile_width = kernel_size
    if len(image.shape) == 3:
        img_height, img_width, channels = image.shape
        tiled_array = image.reshape(img_height // tile_height,
                                    tile_height,
                                    img_width // tile_width,
                                    tile_width,
                                    channels)
        tiled_array = tiled_array.swapaxes(1, 2)
        tiled_array = tiled_array.reshape(-1, kernel_size[0], kernel_size[1], 3)

    elif len(image.shape) == 2:
        img_height, img_width = image.shape
        tiled_array = image.reshape(img_height // tile_height,
                                    tile_height,
                                    img_width // tile_width,
                                    tile_width)

        tiled_array = tiled_array.swapaxes(1, 2)
        tiled_array = tiled_array.reshape(-1, kernel_size[0], kernel_size[1])

    elif len(image.shape) == 1:
        img_height = int(np.sqrt(image.shape[0]))
        img_width = int(np.sqrt(image.shape[0]))
        tiled_array = image.reshape(img_height // tile_height,
                                    tile_height,
                                    img_width // tile_width,
                                    tile_width)
        tiled_array = tiled_array.swapaxes(1, 2)
        tiled_array = tiled_array.reshape(-1, kernel_size[0], kernel_size[1])
    return tiled_array