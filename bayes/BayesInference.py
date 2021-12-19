from collections.abc import Iterable
from itertools import combinations
from utils import simplify_image, get_heatmap, read_data
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os


class BayesInference:
    def __init__(self, disease_categories: dict[str, int] = None, num_of_pixel: int = 100 * 100,
                 age_categories: dict[str: int] = None) -> None:

        if disease_categories is None:
            disease_categories = {'blindness': 0, 'bone cancer': 1}
        if age_categories is None:
            age_categories = {'junior': 0, 'adult': 1, 'senior': 2}

        self.images_data = None
        self.diseases_data = None
        self.age_data = None
        self.diseases_categories = disease_categories
        self.num_of_pixel = num_of_pixel
        self.age_categories = age_categories

    def load_data_with_path(self, image_directory_path: str, age_on_images: str = 'adult', diseases: Iterable[str] = (),
                            load_heatmap: bool = False) -> None:

        data = np.empty((0, self.num_of_pixel), int)
        for file in os.listdir(image_directory_path):
            if load_heatmap:
                single_image_data = get_heatmap(f'{image_directory_path}/{file}')
            else:
                image = Image.open(f'{image_directory_path}/{file}').convert('L')
                single_image_data = np.asarray(image)
            data = np.vstack([data, single_image_data.flatten()])

        self.load_data(images_data=data, age_on_images=age_on_images, diseases_on_images=diseases)

    def load_data(self, images_data: np.ndarray, age_on_images: str = None,
                  diseases_on_images: Iterable[str] = ()) -> None:

        if self.images_data is None:
            self.images_data = np.empty(shape=(0, self.num_of_pixel), dtype=bool)
            self.diseases_data = np.empty(shape=(0, len(self.diseases_categories)))
            self.age_data = np.empty(shape=(0, len(self.age_categories)))

        self.images_data = np.vstack([self.images_data, images_data])
        self._update_disease_data(diseases_on_images=diseases_on_images, amount_of_records=images_data.shape[0])
        self._update_age_data(age_on_images=age_on_images, amount_of_records=images_data.shape[0])

    def _update_age_data(self, age_on_images: str, amount_of_records: int) -> None:
        age_data = np.zeros(shape=(amount_of_records, len(self.age_categories)), dtype=bool)
        age_data[:, self.age_categories[age_on_images]] = True
        self.age_data = np.vstack([self.age_data, age_data])

    def _update_disease_data(self, diseases_on_images: Iterable[str], amount_of_records: int) -> None:
        diseases_data = np.zeros(shape=(amount_of_records, len(self.diseases_categories)), dtype=bool)
        for disease in diseases_on_images:
            diseases_data[:, self.diseases_categories[disease]] = True
        self.diseases_data = np.vstack([self.diseases_data, diseases_data])

    def get_distribution(self, new_image_encoded: np.ndarray, encoded_images_data: np.ndarray,
                         similarity_threshold: float = 0.9) -> float:
        different_pixels = encoded_images_data ^ new_image_encoded
        similarity = 1 - np.sum(different_pixels, axis=1) / encoded_images_data.shape[1]
        similar_photos = similarity > similarity_threshold
        if encoded_images_data.shape[0] > 0:
            return np.sum(similar_photos) / encoded_images_data.shape[0] + 1e-2
        return 1e-2

    def get_diseases_probabilities(self, new_image_path: str, encode_bits_num: int = 8,
                                   similarity_threshold: float = 0.9, pooling_slice_area: int = 1,
                                   pooling_function=np.max):

        new_image_data = np.asarray(Image.open(f'{new_image_path}').convert('L'))
        new_image_pooled = simplify_image(new_image_data, (pooling_slice_area, pooling_slice_area),
                                          pooling_function)
        images_data_pooled = np.apply_along_axis(simplify_image, 1, self.images_data,
                                                 slice_shape=(pooling_slice_area, pooling_slice_area),
                                                 pooling_func=pooling_function)
        new_image_pooled = new_image_pooled.reshape(1, -1)
        images_data_pooled = images_data_pooled.reshape(images_data_pooled.shape[0], -1)

        new_image_encoded_data = self.one_hot_encode_images(data=new_image_pooled, bits_num=encode_bits_num)
        encoded_images_data = self.one_hot_encode_images(data=images_data_pooled, bits_num=encode_bits_num)

        result_probabilities = {}
        denominator = self.get_distribution(new_image_encoded_data, encoded_images_data, similarity_threshold)

        for disease in self._get_all_diseases_combinations():
            name = disease
            indexes = self._get_indexes_by_disease(encoded_images_data, disease_condition=disease)
            prior = len(indexes) / self.diseases_data.shape[0]
            likelihood = self.get_distribution(new_image_encoded_data, encoded_images_data[indexes],
                                               similarity_threshold)
            result_probabilities[name] = prior * likelihood / denominator

        return sorted(result_probabilities.items(), key=lambda item: item[1], reverse=True)

    def one_hot_encode_images(self, data: np.ndarray = None, bits_num: int = 8) -> np.ndarray:
        encoder = OneHotEncoder(categories=[[i for i in range(2 ** bits_num)] for _ in range(data.shape[-1])],
                                dtype=bool, handle_unknown='ignore', sparse=False)
        if data is None:
            data_to_encode = np.floor(self.images_data / (2 ** (8 - bits_num)))
        else:
            data_to_encode = np.floor(data / (2 ** (8 - bits_num)))
        encoded_pixel_features = encoder.fit_transform(data_to_encode)
        return encoded_pixel_features

    def _get_indexes_by_disease(self, data: np.ndarray, disease_condition: Iterable[str] = ()) -> list[int]:
        indexes = set(range(data.shape[0]))
        for disease in disease_condition:
            current_disease_indexes = set(*np.where(self.diseases_data[:, self.diseases_categories[disease]] == 1))
            indexes = indexes.intersection(current_disease_indexes)
        return list(indexes)

    def _get_all_diseases_combinations(self) -> list[tuple[str]]:
        disease_combinations = []
        for i in range(len(self.diseases_categories)):
            disease_combinations += combinations(self.diseases_categories, i + 1)
        return disease_combinations


if __name__ == '__main__':

    input_data_paths = read_data('/Users/hwdowiak/Desktop/inzynierka/data/final_photos')
    all_diseases = set()
    for dir in input_data_paths:
        for disease in dir[1]:
            all_diseases.add(disease)
    enumerated_diseases = {disease: i for i, disease in enumerate(all_diseases)}
    bi = BayesInference(disease_categories=enumerated_diseases, age_categories={'junior': 0, 'adult': 1, 'senior': 2})

    for path, disease, age_category in input_data_paths:
        bi.load_data_with_path(image_directory_path=path, age_on_images=age_category, diseases=disease, load_heatmap=True)

    new_img_path = '/Users/hwdowiak/Desktop/inzynierka/data/test/Tucker-1-e1639748696180.jpeg'

    result = bi.get_diseases_probabilities(new_image_path=new_img_path,
                                           encode_bits_num=2,
                                           similarity_threshold=0.9,
                                           pooling_slice_area=2,
                                           pooling_function=np.max)
    print(result)
