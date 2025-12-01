import io
import torch
import numpy as np
from PIL import Image
from typing import Union, List
from transformers import pipeline


class EmbeddingExtractor:
    """
    A class for extracting embeddings from images using pre-trained foundational models.

    Attributes:
        return_type (str): The type of the returned embeddings. Defaults to 'numpy'.
        extractor (pipeline): The pipeline for image feature extraction.
    """

    def __init__(
        self,
        model: str = "facebook/dinov2-base",
        batch_size: int = 1,
        num_workers: int = 8,
        return_type: str = "numpy",
        device: str = "cpu",
    ):
        """
        Initializes the EmbeddingExtractor with a specified model, device, and return type.

        Args:
            model (str, optional): The model to use for feature extraction. Defaults to 'facebook/dinov2-base'.
            batch_size (int, optional): The size of the batch to use for feature extraction. Defaults to 1.
            num_workers (int, optional): The number of workers to use for feature extraction. Defaults to 8.
            return_type (str, optional): The type of the returned embeddings. Defaults to 'numpy'.
            device (str, optional): The device to use for computation. Defaults to 'cpu'.
        """

        if model not in ["facebook/dinov2-base", "google/vit-base-patch16-224-in21k"]:
            raise ValueError(
                "Unsupported model. Only 'facebook/dinov2-base' and 'google/vit-base-patch16-224-in21k' are supported."
            )

        self.return_type = return_type
        if self.return_type not in ["numpy", "tensor"]:
            raise ValueError("Unsupported return type. Only 'numpy' and 'tensor' are supported.")

        self.extractor = pipeline(
            model=model,
            task="image-feature-extraction",
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            pool=True,
            framework="pt",
        )

    def extract(
        self,
        images: Union[str, List[str], bytes, List[bytes], Image.Image, List[Image.Image]],
        timeout: float = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Extracts the features of the input(s).

        Args:
            images (`str`, `List[str]`, `bytes`, `List[bytes]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded as bytes
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, all as image bytes or all as PIL
                images.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is used and
                the call may block forever.
        Returns:
            A nested tensor or numpy array of features computed by the model is returned depending on what the class was initialized with.
        """
        if isinstance(images, bytes):
            images = Image.open(io.BytesIO(images))
        elif isinstance(images, list):
            images = [
                Image.open(io.BytesIO(img)) if isinstance(img, bytes) else img for img in images
            ]

        embeddings = self.extractor(images, timeout=timeout, return_tensors=True)
        if isinstance(embeddings, list):
            embeddings = torch.concat(embeddings, dim=0)

        if self.return_type == "numpy":
            return embeddings.cpu().numpy()
        else:
            return embeddings