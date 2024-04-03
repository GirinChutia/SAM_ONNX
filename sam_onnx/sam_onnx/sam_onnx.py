from copy import deepcopy
from typing import Any, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import glob


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
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
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


class SamEncoder:
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
        feature = self.session.run(None, {self.input_name: tensor})[0]
        return feature

    def __call__(self, img: np.array, *args: Any, **kwds: Any) -> Any:
        return self._extract_feature(img)


class SamDecoder:
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
    ):
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
        low_res_masks, iou_predictions = self.session.run(None, input_dict)
        masks = np_mask_postprocessing(low_res_masks, np.array(origin_image_size))

        if not return_logits:
            masks = masks > self.mask_threshold
        return masks, iou_predictions, low_res_masks

    def apply_coords(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes


def np_resize_longest_image_size(
    input_image_size: np.array, longest_side: int
) -> np.array:
    scale = longest_side / np.max(input_image_size)
    transformed_size = scale * input_image_size
    transformed_size = np.floor(transformed_size + 0.5).astype(np.int64)
    return transformed_size


def np_interp(x: np.array, size: tuple):
    _rmsk = []
    for m in range(x.shape[0]):
        msk = x[m, 0, :, :]
        resized_array = cv2.resize(msk, size, interpolation=cv2.INTER_LINEAR)
        _rmsk.append(resized_array)
    np_rmsk = np.array(_rmsk)
    np_rmsk = np_rmsk[:, np.newaxis, :, :]
    return np_rmsk


def np_mask_postprocessing(masks: np.array, orig_im_size: np.array) -> np.array:

    img_size = 1024
    masks = np_interp(masks, (img_size, img_size))
    
    prepadded_size = np_resize_longest_image_size(orig_im_size, img_size)
    masks = masks[..., : int(prepadded_size[0]), : int(prepadded_size[1])]

    origin_image_size = orig_im_size.astype(np.int64)
    w, h = origin_image_size[0], origin_image_size[1]
    masks = np_interp(masks, (h, w))
    return masks

def preprocess_np(x, img_size):
    pixel_mean = np.array([123.675 / 255, 116.28 / 255, 103.53 / 255]).astype(
        np.float32
    )
    pixel_std = np.array([58.395 / 255, 57.12 / 255, 57.375 / 255]).astype(np.float32)

    oh, ow, _ = x.shape
    long_side = max(oh, ow)
    if long_side != img_size:
        scale = img_size * 1.0 / max(oh, ow)
        newh, neww = int(oh * scale + 0.5), int(ow * scale + 0.5)
        x = cv2.resize(x, (neww, newh))

    h, w = x.shape[:2]
    x = x.astype(np.float32) / 255
    x = (x - pixel_mean) / pixel_std
    th, tw = img_size, img_size
    assert th >= h and tw >= w

    x = np.pad(
        x,
        ((0, th - h), (0, tw - w), (0, 0)),
        mode="constant",
        constant_values=0,  # (top, bottom), (left, right)
    ).astype(np.float32)
    x = x.transpose((2, 0, 1))[np.newaxis, :, :, :]

    return x
            
class InferSAM:
    def __init__(self, model_dir, model_name="l0"):
        assert model_dir is not None, "model_dir is null"
        assert model_name is not None, "model_name is null"

        self.model_name = model_name
        self.encoder = SamEncoder(glob.glob(model_dir + "/*_encoder.onnx")[0])
        self.decoder = SamDecoder(glob.glob(model_dir + "/*_decoder.onnx")[0])

    def infer(
        self, img_path, boxes: list[list] = [[80, 50, 320, 420], [300, 20, 530, 420]]
    ):
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
        
        return masks
