import argparse
import typing
from abc import ABC, abstractmethod
from typing import Callable, List

import cv2
import numpy as np
import pycuda.autoinit  # pylint: disable=unused-import
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


class MBV4Preprocessor:
    """Preprocessor for MobileNetV4 models using ONNX Runtime format."""

    def __init__(self, input_size: typing.Tuple[int, int], interpolation: int = 2):
        """
        Initializes the MBV4Preprocessor class.

        :param input_size: The input size of the model. (h, w)
        :param interpolation: The interpolation method to use (default is 2).
        """
        self.input_size = input_size
        self.interpolation = interpolation

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transforms the input image array.

        :param x: The input image array.
        :return: The transformed image array.
        """
        # Convert BGR to RGB
        x = x[..., ::-1]  # Flip the color channels (from BGR to RGB)

        # Resize the image to the input size (using numpy)
        # We will use PIL for resizing as numpy doesn't have built-in support for resizing
        img = Image.fromarray(x)
        img = img.resize(
            (self.input_size[1], self.input_size[0]), resample=self.interpolation
        )  # (w, h)
        # img = img.resize((298, 168), resample=Image.BILINEAR)  # 否則維持原本的 resize

        # Convert back to numpy array
        x = np.array(img)

        # Apply normalization (you can customize the mean and std values here)
        mean = np.array([0.485, 0.456, 0.406])  # Example ImageNet means
        std = np.array([0.229, 0.224, 0.225])  # Example ImageNet stds
        x = (
            x / 255.0 - mean
        ) / std  # Normalize to range [0, 1], then apply mean/std normalization

        # 轉換為 HWC 格式並保存
        x_vis = (x * std + mean) * 255  # 還原到原始範圍
        x_vis = np.clip(x_vis, 0, 255).astype(np.uint8)
        cv2.imwrite("python_preprocessed.png", cv2.cvtColor(x_vis, cv2.COLOR_RGB2BGR))

        x = np.transpose(x, (2, 0, 1))  # HWC to CHW
        return x

    # Output shape: (C, H, W)
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply the transformation to input data and return a numpy array.

        :param data: The input data to be transformed.
        :return: The transformed data as a numpy array.
        """
        transformed = self.transform(data)
        return transformed.astype(np.float32)


class BaseModel(ABC):
    """Base class for inference models"""

    def __init__(
        self,
        model_path: str,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = None,
        postprocess_fn: Callable[[np.ndarray], np.ndarray] = None,
    ):
        """Initialize the Model object.

        :param model_path: The path to the model.
        :param preprocess_fn: A function to preprocess the input data (optional).
        :param postprocess_fn: A function to postprocess the output data (optional).
        """
        self.model_path = model_path
        self.preprocess_fn = preprocess_fn or (lambda x: x)
        self.postprocess_fn = postprocess_fn or (lambda x: x)

    @abstractmethod
    def load_model(self) -> None:
        """Load the model."""

    @abstractmethod
    def get_batch_size(self) -> int:
        """Get batch size."""

    def run_inference(self, inputs: List[np.ndarray]) -> np.ndarray:
        """Default inference method, subclasses can choose to override it.

        :param inputs: A list of input arrays.
        :return: The result of the inference.
        """
        processed_inputs = self.preprocess_fn(inputs)
        result = self._run_inference(processed_inputs)
        return self.postprocess_fn(result)

    @abstractmethod
    def _run_inference(self, input_tensors: List[np.ndarray]) -> np.ndarray:
        """Subclasses should implement the specific inference logic"""


class TensorRTModel(BaseModel):
    """Inference class for TensorRT models."""

    def __init__(
        self,
        engine_path: str,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = None,
        postprocess_fn: Callable[[np.ndarray], np.ndarray] = None,
    ):
        """Initializes the TensorRTModel class.

        :param engine_path: The path to the TensorRT engine file.
        :param preprocess_fn: A function to preprocess the input data. Defaults to None.
        :param postprocess_fn: A function to postprocess the output data. Defaults to None.
        """
        super().__init__(engine_path, preprocess_fn, postprocess_fn)
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.engine = None
        self.context = None
        self.stream = None
        self.input_memory = None
        self.output_memory = None
        self.output_size = None
        self.load_model()

    def load_model(self) -> None:
        """Load the TensorRT engine and initialize necessary resources."""
        runtime = trt.Runtime(self.logger)

        # Read the engine file into a memory buffer and deserialize it
        with open(self.model_path, "rb") as f:
            model_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(model_data)

        # Additional state for intermediate activations for inference
        self.context = self.engine.create_execution_context()

        # Get input and output sizes
        input_tensor_name = None
        output_tensor_name = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_tensor_name = name
            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                output_tensor_name = name
                self.output_size = np.prod(self.engine.get_tensor_shape(name))

        if input_tensor_name is None or output_tensor_name is None:
            raise ValueError("Tensor names for input/output could not be resolved.")

    def allocate_memory(self, input_data: np.ndarray) -> None:
        """Allocates memory for input and output data on the device.

        :param input_data: The input data to be allocated on the device.
        :return: None.
        """
        # Check the data type and shape
        target_dtype = input_data.dtype

        # Initialize output memory
        output_mem = np.empty(self.output_size, dtype=target_dtype)

        # Allocate device memory
        self.input_memory = cuda.mem_alloc(input_data.nbytes)
        self.output_memory = cuda.mem_alloc(output_mem.nbytes)

        # Verify tensor names and count
        tensor_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        if len(tensor_names) != 2:
            raise ValueError(
                "Expected 2 tensor names, but found {0}".format(len(tensor_names))
            )

        # Set tensor addresses
        self.context.set_tensor_address(tensor_names[0], int(self.input_memory))
        self.context.set_tensor_address(tensor_names[1], int(self.output_memory))

        self.stream = cuda.Stream()

    def get_batch_size(self) -> int:
        """Get batch size from the engine's input tensor shape."""
        input_shape = self.engine.get_tensor_shape(self.engine.get_tensor_name(0))
        batch_size = input_shape[0]  # The first dimension is the batch size
        return batch_size

    def _run_inference(self, input_tensors: List[np.ndarray]) -> np.ndarray:
        """Run inference with inputs and optional postprocessing.

        :param input_tensors: List of input tensors for inference.
        :return: Output tensor(s) from the inference.
        """
        if len(input_tensors) != 1:
            raise ValueError("TensorRT engine expects a single input tensor.")

        # input_data = input_tensors[0].astype(np.float32)
        input_data = np.ascontiguousarray(input_tensors[0].astype(np.float32))

        if self.stream is None:
            self.allocate_memory(input_data)

        # Transfer input data to device
        cuda.memcpy_htod_async(self.input_memory, input_data, self.stream)

        # Execute the model
        self.context.execute_async_v3(self.stream.handle)

        # Allocate memory for output and transfer predictions back
        output_data = np.empty(self.output_size, dtype=np.float32)
        cuda.memcpy_dtoh_async(output_data, self.output_memory, self.stream)

        # Synchronize threads
        self.stream.synchronize()

        reshaped_output = output_data.reshape(1, -1)
        final_output = [np.array(reshaped_output, dtype=np.float32)]
        return final_output


def main():
    parser = argparse.ArgumentParser(description="Test an ONNX model.")
    parser.add_argument(
        "--ann", type=str, required=True, help="Path to the annotation file."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the ONNX model."
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,  # Allows one or two values
        required=True,
        help="Input size of the model. Use one value for square (e.g., 224) or two for (height, width) (e.g., 360 640).",
    )
    args = parser.parse_args()

    model = TensorRTModel(args.model)

    all_predictions = []
    all_labels = []

    preprocess = MBV4Preprocessor(args.input_size)

    import time
    total_time = 0

    with open(args.ann, "r") as f:
        for index, line in enumerate(f):
            image_path, label = line.strip().split()
            image = cv2.imread(image_path)
            image = preprocess(image)

            image = np.expand_dims(image, axis=0)  # Add batch dimension

            t0 = time.time()
            prediction = model.run_inference([image])
            total_time += time.time() - t0

            print(index, ":", prediction[0], label + " " * 50, end="\r")

            all_predictions.append(prediction[0])
            all_labels.append(np.array([int(label)]))

    print()
    print(f"time elapsed: {total_time:.4f} seconds")

    all_labels = np.concatenate(all_labels)  # Flatten to 1D
    all_pred_labels = np.argmax(
        np.concatenate(all_predictions), axis=1
    )  # (N, C) -> (N,)
    # Filter out dummy data (-1 labels)
    valid_indices = all_labels != -1

    # Generate classification report
    report = classification_report(
        all_labels[valid_indices], all_pred_labels[valid_indices]
    )

    # Generate confusion matrix
    cm = confusion_matrix(all_labels[valid_indices], all_pred_labels[valid_indices])

    print(" " * 80)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    main()
