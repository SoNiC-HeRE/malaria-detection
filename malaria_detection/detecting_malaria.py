import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model(r'C:\Users\sriya\Downloads\malaria_detection_project\malaria_detection_project\malaria_detection\malaria_parasite_detection_model_2.h5')


optimizer = tf.keras.optimizers.Adam()  # Use the same optimizer as before
loss = 'binary_crossentropy'  # Use the same loss function as before
metrics = ['accuracy']  # Add any additional metrics if needed

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def detect_malaria_parasites(image):
    # Convert image bytes to numpy array
    nparr = np.frombuffer(image.read(), np.uint8)
    # Decode image
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_np is None:
        print("Error: Unable to load the image.")
        return None, None

    if image_np.size == 0:
        print("Error: Empty image detected.")
        return None, None

    # Resize 
    resized_image = cv2.resize(image_np, (224, 224))
    preprocessed_image = resized_image / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    # Predict
    predictions = model.predict(preprocessed_image)

    if predictions.shape[1] != 2:
        print("Error: Unexpected shape of predictions. Expected shape: (1, 2), Actual shape:", predictions.shape)
        return None, None

    # probability
    parasitized_prob, uninfected_prob = predictions[0]

    if parasitized_prob > uninfected_prob:
        class_label = 'Parasitized'
    else:
        class_label = 'Uninfected'

    return class_label, parasitized_prob
