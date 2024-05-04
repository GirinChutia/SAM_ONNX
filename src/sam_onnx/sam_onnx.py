from copy import deepcopy
from typing import Any, Tuple, Union, List
import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import glob
import gdown
import os

def check_and_download_weights(model_name='l0'):

    __supported_modelnames = ['l0', 'xl0']

    assert model_name in __supported_modelnames, f'Model name not supported. Please use one of : {__supported_modelnames}'
    
    l0_weights = {'encoder' : 'https://drive.google.com/file/d/1a0tRmHQeGTAbSeMqBMhu4DinsOR3cSv6/view?usp=sharing',
                 'decoder': 'https://drive.google.com/file/d/13J7pNfh016sBqOQ17CludkUFdKgkkyQM/view?usp=sharing'}
    
    xl0_weights = {'encoder': 'https://drive.google.com/file/d/1NzavgCAqk6mSzTnQ_LKfl78V_O68lWNX/view?usp=sharing',
                  'decoder': 'https://drive.google.com/file/d/1lrn5bQRE01Mwtp-nr9DBNTHcxk4Q6iiP/view?usp=sharing'}
    
    if os.path.exists('model_weights'):
        model_weights_folder_path = os.path.abspath('model_weights')
    else:
        os.makedirs('model_weights', 
                    exist_ok = False)
        model_weights_folder_path = os.path.abspath('model_weights')
    
    if os.path.exists(f'model_weights/{model_name}/encoder.onnx'):
        encoder_weights_path = os.path.abspath(f'model_weights/{model_name}/encoder.onnx')
    else:
        os.makedirs(f'model_weights/{model_name}', 
                    exist_ok = True)
        if model_name == 'l0':
            gdown.download(l0_weights['encoder'],
                           f'model_weights/{model_name}/encoder.onnx', 
                           fuzzy=True)
        if model_name == 'xl0':
            gdown.download(xl0_weights['encoder'],
                           f'model_weights/{model_name}/encoder.onnx', 
                           fuzzy=True)
        encoder_weights_path = os.path.abspath(f'model_weights/{model_name}/encoder.onnx')
            
    if os.path.exists(f'model_weights/{model_name}/decoder.onnx'):
        decoder_weights_path = os.path.abspath(f'model_weights/{model_name}/decoder.onnx')
    else:
        if model_name == 'l0':
            gdown.download(l0_weights['decoder'],
                           f'model_weights/{model_name}/decoder.onnx', 
                           fuzzy=True)
        if model_name == 'xl0':
            gdown.download(xl0_weights['decoder'],
                           f'model_weights/{model_name}/decoder.onnx', 
                           fuzzy=True)
        decoder_weights_path = os.path.abspath(f'model_weights/{model_name}/decoder.onnx')

    return encoder_weights_path, decoder_weights_path
    
def show_mask(mask, ax, random_color=False):
    """
    Visualize a mask image on the given axis.

    Parameters
    ----------
    mask : np.ndarray
        The mask image to visualize.
    ax : matplotlib.axes.Axes
        The axis to plot on.
    random_color : bool, optional
        Whether to use a random color for the mask, by default False
    """
    if random_color:
        # Create a random color with some transparency
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # Use a specific color with some transparency
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """
    Show points on the axis.

    Parameters
    ----------
    coords : np.ndarray
        The coordinates of the points to show.
    labels : np.ndarray
        The labels of the points.
    ax : matplotlib.axes.Axes
        The axis to plot on.
    marker_size : int, optional
        The size of the markers, by default 375
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    """
    Show a bounding box on the axis.

    Parameters
    ----------
    box : list
        The bounding box coordinates as [x0, y0, x1, y1].
    ax : matplotlib.axes.Axes
        The axis to plot on.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


class SamEncoder:
    """
    The encoder class that loads and runs the SAM encoder model.

    Parameters
    ----------
    model_path: str
        The path to the encoder model.
    device: str, optional (default is 'cpu')
        The device to run the model, either 'cuda' or 'cpu'.
    kwargs: dict
        Additional arguments to be passed to the `InferenceSession` class from
        the onnxruntime library.

    Attributes
    ----------
    session: InferenceSession
        The loaded encoder model.
    input_name: str
        The name of the input layer of the model.
    """

    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        print(f"loading encoder model from {model_path}...")
        self.session = ort.InferenceSession(
            model_path, opt, providers=provider, **kwargs
        )
        self.input_name = self.session.get_inputs()[0].name

    def _extract_feature(self, tensor: np.ndarray) -> np.ndarray:
        """
        Extract the feature from the input image tensor using the loaded
        encoder model.

        Parameters
        ----------
        tensor: numpy.ndarray
            The input image tensor.

        Returns
        -------
        feature: numpy.ndarray
            The feature extracted from the input image.
        """
        feature = self.session.run(None, {self.input_name: tensor})[0]
        return feature

    def __call__(self, img: np.array, *args: Any, **kwds: Any) -> Any:
        """
        Call the encoder with the input image.

        Parameters
        ----------
        img: numpy.ndarray
            The input image.
        args, kwargs:
            Additional positional and keyword arguments to be passed to the
            encoder.

        Returns
        -------
        feature: numpy.ndarray
            The feature extracted from the input image.
        """
        return self._extract_feature(img)


class SamDecoder:
    """
    The decoder class that loads and runs the SAM decoder model.

    Parameters
    ----------
    model_path: str
        The path to the decoder model.
    device: str, default="cpu"
        The device to run the model, either "cuda" or "cpu".
    target_size: int, default=1024
        The target size of the output mask. The final mask size may be
        smaller if the original image is too small.
    mask_threshold: float, default=0.0
        The threshold value to binarize the output mask.
    kwargs: Any
        Additional arguments to be passed to onnxruntime.InferenceSession.

    Attributes
    ----------
    target_size: int
        The target size of the output mask.
    mask_threshold: float
        The threshold value to binarize the output mask.
    session: onnxruntime.InferenceSession
        The inference session of the loaded decoder model.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        target_size: int = 1024,
        mask_threshold: float = 0.0,
        **kwargs,
    ):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        print(f"loading decoder model from {model_path}...")
        self.target_size = target_size
        self.mask_threshold = mask_threshold
        self.session = ort.InferenceSession(
            model_path, opt, providers=provider, **kwargs
        )

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.

        Parameters
        ----------
        oldh: int
            The height of the input image.
        oldw: int
            The width of the input image.
        long_side_length: int
            The target long side length of the output image.

        Returns
        -------
        Tuple[int, int]
            The (height, width) of the output image after resizing.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def run(
        self,
        img_embeddings: np.ndarray,
        origin_image_size: Union[list, tuple],
        point_coords: Union[list, np.ndarray] = None,
        point_labels: Union[list, np.ndarray] = None,
        boxes: Union[list, np.ndarray] = None,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, Any, Any]:
        """
        Run the SAM decoder to segment an input image.

        Parameters
        ----------
        img_embeddings: np.ndarray
            The image embeddings obtained from SAM encoder.
            The shape should be (1, 256, 64, 64).
        origin_image_size: Union[list, tuple]
            The original size of the input image, (height, width)
        point_coords: Union[list, np.ndarray], optional
            The coordinates of the points in the input image.
            The shape should be (N, 2), where N is the number of points.
        point_labels: Union[list, np.ndarray], optional
            The labels of the points.
            The shape should be (N,) where N is the number of points.
        boxes: Union[list, np.ndarray], optional
            The coordinates of the bounding boxes in the input image.
            The shape should be (M, 4), where M is the number of boxes.
        return_logits: bool, default False
            Whether to return the logits (before sigmoid) of the mask predictions.

        Returns
        -------
        Tuple[np.ndarray, Any, Any]
            The segmentation masks, IoU scores, and low-resolution masks.
        """
        input_size = self.get_preprocess_shape(
            *origin_image_size, long_side_length=self.target_size
        )

        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError(
                "Unable to segment, please input at least one box or point."
            )

        if img_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong embedding shape!")

        if point_coords is not None:
            point_coords = self.apply_coords(
                point_coords, origin_image_size, input_size
            ).astype(np.float32)

            prompts, labels = point_coords, point_labels

        if boxes is not None:
            boxes = self.apply_boxes(boxes, origin_image_size, input_size).astype(
                np.float32
            )
            box_labels = np.array(
                [[2, 3] for _ in range(boxes.shape[0])], dtype=np.float32
            ).reshape((-1, 2))

            if point_coords is not None:
                prompts = np.concatenate([prompts, boxes], axis=1)
                labels = np.concatenate([labels, box_labels], axis=1)
            else:
                prompts, labels = boxes, box_labels

        input_dict = {
            "image_embeddings": img_embeddings,
            "point_coords": prompts,
            "point_labels": labels,
        }

        # Run the inference
        low_res_masks, iou_predictions = self.session.run(None, input_dict)

        # Post-process the masks
        masks = np_mask_postprocessing(low_res_masks, np.array(origin_image_size))

        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def apply_coords(self, coords, original_size, new_size):
        """
        Applies the resizing to the coordinates.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates to be resized.
            The shape should be (N, 2), where N is the number of points.
        original_size : Union[list, tuple]
            The original size of the input image, (height, width)
        new_size : Union[list, tuple]
            The new size of the input image, (height, width)

        Returns
        -------
        np.ndarray
            The resized coordinates.
        """
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        """
        Applies the resizing to the bounding boxes.

        Parameters
        ----------
        boxes : np.ndarray
            The coordinates of the bounding boxes in the input image.
            The shape should be (M, 4), where M is the number of boxes.
        original_size : Union[list, tuple]
            The original size of the input image, (height, width)
        new_size : Union[list, tuple]
            The new size of the input image, (height, width)

        Returns
        -------
        np.ndarray
            The resized bounding boxes.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes


def np_resize_longest_image_size(
    input_image_size: np.array, longest_side: int
) -> np.array:
    """Resizes the image size to the longest side.

    Parameters
    ----------
    input_image_size : np.array
        Size of the input image in (height, width) format.
    longest_side : int
        Desired longest side of the resized image.

    Returns
    -------
    np.array
        Size of the resized image in (height, width) format.
    """
    scale = longest_side / np.max(input_image_size)
    transformed_size = scale * input_image_size
    transformed_size = np.floor(transformed_size + 0.5).astype(np.int64)
    return transformed_size


def np_interp(x: np.array, size: tuple) -> np.array:
    """Interpolates a batch of masks to a given size.

    Parameters
    ----------
    x : np.array
        A batch of masks with shape (batch_size, 1, height, width).
    size : tuple
        Desired size of the masks (height, width) format.

    Returns
    -------
    np.array
        A batch of interpolated masks with shape (batch_size, 1, height, width).
    """
    _rmsk = []
    for m in range(x.shape[0]):
        msk = x[m, 0, :, :]
        resized_array = cv2.resize(msk, size, interpolation=cv2.INTER_LINEAR)
        _rmsk.append(resized_array)
    np_rmsk = np.array(_rmsk)
    np_rmsk = np_rmsk[:, np.newaxis, :, :]
    return np_rmsk


def np_mask_postprocessing(masks: np.array, orig_im_size: np.array) -> np.array:
    """
    Perform postprocessing on predicted masks by interpolating them to
    desired size and then resizing them back to original image size.

    Parameters
    ----------
    masks : np.array
        Predicted masks.
    orig_im_size : np.array
        Original image size.

    Returns
    -------
    np.array
        Postprocessed masks.
    """

    img_size = 1024  # Desired output size
    masks = np_interp(masks, (img_size, img_size))

    # Pad predicted masks to desired output size
    prepadded_size = np_resize_longest_image_size(orig_im_size, img_size)
    masks = masks[..., : int(prepadded_size[0]), : int(prepadded_size[1])]

    # Resize padded masks back to original image size
    origin_image_size = orig_im_size.astype(np.int64)
    w, h = origin_image_size[0], origin_image_size[1]
    masks = np_interp(masks, (h, w))
    return masks

def preprocess_np(x, img_size):
    """
    Preprocess an image with mean and std normalization and padding to
    desired size.

    Parameters
    ----------
    x : numpy.ndarray
        Image to be preprocessed.
    img_size : int
        Desired size of the longer edge of the image.

    Returns
    -------
    numpy.ndarray
        Preprocessed image.

    """
    pixel_mean = np.array([123.675 / 255, 116.28 / 255, 103.53 / 255]).astype(np.float32)
    pixel_std = np.array([58.395 / 255, 57.12 / 255, 57.375 / 255]).astype(np.float32)

    oh, ow, _ = x.shape
    long_side = max(oh, ow)
    if long_side != img_size:
        # Resize the image with long side == img_size
        scale = img_size * 1.0 / max(oh, ow)
        newh, neww = int(oh * scale + 0.5), int(ow * scale + 0.5)
        x = cv2.resize(x, (neww, newh))

    h, w = x.shape[:2]
    x = x.astype(np.float32) / 255  # Normalize to [0, 1]
    x = (x - pixel_mean) / pixel_std  # Normalize pixel values
    th, tw = img_size, img_size
    assert th >= h and tw >= w, "image is too small"

    # Pad the image with zeros if shorter than desired size
    x = np.pad(
        x,
        ((0, th - h), (0, tw - w), (0, 0)),
        mode="constant",
        constant_values=0,  # (top, bottom), (left, right)
    ).astype(np.float32)

    # Transpose the image from HWC to CHW and add batch dimension
    x = x.transpose((2, 0, 1))[np.newaxis, :, :, :]

    return x
            
class InferSAM:
    """
    Class for inference with SAM models.

    Parameters
    ----------
    model_dir : str
        Directory containing trained SAM model.
    model_name : str, default 'l0'
        Name of the model to use.
        Must be one of ['l0', 'l1', 'l2', 'xl0', 'xl1'].

    Attributes
    ----------
    model_name : str
        Name of the model to use.
    encoder : SamEncoder
        The encoder part of the SAM model.
    decoder : SamDecoder
        The decoder part of the SAM model.

    """

    def __init__(self, model_name: str = "l0"):
        # assert model_dir is not None, "model_dir is null"
        assert model_name is not None, "model_name is null"

        self.model_name = model_name
        encoder_weights_path, decoder_weights_path = check_and_download_weights(model_name)
        
        # Find encoder and decoder models
        encoder_path = encoder_weights_path # glob.glob(model_dir + "/*_encoder.onnx")[0]
        decoder_path = decoder_weights_path # glob.glob(model_dir + "/*_decoder.onnx")[0]

        self.encoder = SamEncoder(encoder_path)
        self.decoder = SamDecoder(decoder_path)

        self.figsize = (10,10)

    def infer(
        self,
        img_path: str,
        boxes: List[list] = [[80, 50, 320, 420], [300, 20, 530, 420]],
        visualize=False,
    ) -> np.array:
        """
        Infer segmentation masks for a given image using the SAM model.

        Parameters
        ----------
        img_path : str
            Path to the input image.
        boxes : list of lists, default [[80, 50, 320, 420], [300, 20, 530, 420]]
            List of boxes, each box is a list of 4 ints, representing
            [xmax, ymax, xmin, ymin] coordinates.

        Returns
        -------
        masks : np.array
            A numpy array of shape (N, 1, H, W) containing segmentation masks,
            where N is the number of boxes, H and W are the height and width of
            the input image.

        """
        assert img_path is not None, "img_path is null"

        raw_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        assert raw_img is not None, "raw_img is null"

        origin_image_size = raw_img.shape[:2]

        img = None
        if self.model_name in ["l0", "l1", "l2"]:
            img = preprocess_np(raw_img, img_size=512)
        elif self.model_name in ["xl0", "xl1"]:
            img = preprocess_np(raw_img, img_size=1024)
        assert img is not None, "img is null"

        boxes = np.array(boxes, dtype=np.float32)  # xmax, ymax, xmin, ymin

        img_embeddings = self.encoder(img)
        masks, _, _ = self.decoder.run(
            img_embeddings=img_embeddings,
            origin_image_size=origin_image_size,
            boxes=boxes,
        )
        if visualize:
            plt.figure(figsize=self.figsize)
            plt.imshow(raw_img)
            for mask in masks:
                show_mask(mask, plt.gca(), 
                          random_color=True)
            for box in boxes:
                show_box(box, plt.gca())
            plt.show()
        return masks

    def set_figsize(self,figsize=(10,10)):
        self.figsize = figsize


