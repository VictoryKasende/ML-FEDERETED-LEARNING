import flwr as fl
import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.model = self.create_model()

    def get_parameters(self, config):
        print("\n\n\n---------------- Get Parameters -----------------")
        return self.model.get_weights()

    def fit(self, parameters, config):
        print("\n\n\n----------------  Train ----------------- ")
        self.model.set_weights(parameters)
        train_data, train_target, _, _ = self.load_data()
        history = self.model.fit(train_data, train_target, epochs=1, verbose=0)
        print("Training loss:", history.history['loss'][0])
        return self.model.get_weights(), len(train_data), {"loss": history.history['loss'][0]}

    def evaluate(self, parameters, config):
        print("\n\n\n----------------  Test ----------------- ")
        self.model.set_weights(parameters)
        _, _, test_data, test_target = self.load_data()
        loss, accuracy = self.model.evaluate(test_data, test_target, verbose=0)
        print("Test loss:", loss)
        print("Test accuracy:", accuracy)
        return loss, len(test_data), {"accuracy": accuracy}

    def load_data(self):
        img_size = 100
        data = []
        target = []

        categories = os.listdir(self.data_dir)
        label_dict = {category: i for i, category in enumerate(categories)}

        for category in categories:
            folder_path = os.path.join(self.data_dir, category)
            img_names = os.listdir(folder_path)
            for img_name in img_names:
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                try:
                    resized = cv2.resize(img, (img_size, img_size))
                    data.append(resized)
                    target.append(label_dict[category])
                except Exception as e:
                    print('Exception:', e)

        data = np.array(data) / 255.0
        data = np.reshape(data, (data.shape[0], img_size, img_size, 3))
        target = np.array(target)

        # Split data into train and test sets
        train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1)

        # Convert target labels to one-hot encoded format
        train_target = tf.keras.utils.to_categorical(train_target)
        test_target = tf.keras.utils.to_categorical(test_target)

        return train_data, train_target, test_data, test_target

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print("Model created")
        return model

# Define the directory containing the dataset
data_dir = '/home/victory/Documents/ML/TP_2/datasets/binome_4'

client = FlowerClient(data_dir)
fl.client.start_numpy_client(
    server_address="localhost:8080",  # Replace with the address of your central server
    client=client,
    grpc_max_message_length=1024*1024*1024
)

