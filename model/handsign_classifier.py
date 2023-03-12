import numpy as np
import tensorflow as tf


class HandSignClassifier:
    def __init__(self, model_path='model/handsign_classifier-model.tflite', num_threads=1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmarks):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_details_tensor_index, np.array([landmarks], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        max_idx = np.argmax(np.squeeze(result))

        prediction = result[0][max_idx]
        if prediction < 0.25:
            return None

        return max_idx
