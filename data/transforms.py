import numpy as np
import torch
import copy

from PIL import Image
from torchvision.transforms import Resize, Normalize

from skimage.measure import regionprops, label
from skimage.filters import threshold_li


def load_image(image_path, size, canvas):

    if "CXR" in image_path:
        if "dcm" in image_path or "dicom" in image_path:
            import pydicom
            dicom = pydicom.read_file(image_path)
            img = np.array(dicom.pixel_array, dtype=float)
            if np.max(img) > 255 or np.min(img) < 0:
                img = (img - np.min(img)) / (np.max(img - np.min(img)) / 255.0)
            img = np.expand_dims(img, 0)
        else:
            img = Image.open(image_path)
            max_size = max(img.size)
            scale = max_size / size[0]
            img.draft('L', (img.size[0] / scale, img.size[1] // scale))
            img = np.asarray(img, dtype=float)

            if (len(img.shape) > 2) and (img.shape[-1] < 5):
                img = img[:, :, 0]
            img = np.expand_dims(img, 0)

    elif "Ophthalmology" in image_path or "HISTOLOGY" in image_path:
        img = Image.open(image_path.replace("é", "é")).convert('RGB')
        img = np.asarray(img, dtype=float)

        img = np.transpose(img, (2, 0, 1))

    img /= 255.

    if "Ophthalmology" in image_path:
        img = crop_im(img)

    img = torch.tensor(img)
    if not canvas or (img.shape[-1] == img.shape[-2]):
        img = Resize(size)(img)
    else:
        sizes = img.shape[-2:]
        max_size = max(sizes)
        scale = max_size / size[0]
        img = Resize((int(img.shape[-2] / scale), int((img.shape[-1] / scale)))).cuda()(img.cuda())
        img = torch.nn.functional.pad(img,
                                      (0, size[0] - img.shape[-1], 0, size[1] - img.shape[-2], 0, 0))
    img = img.cpu().numpy()
    return img


def norm_image(img, norm):
    if img.shape[0] == 1:
        img = np.repeat(img, 3, 0)

    if norm:
        img = torch.tensor(img)
        img = Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])(img)
        img = img.numpy()

    return img


class LoadImage():

    def __init__(self, size=(224, 224), canvas=True, norm=False):
        self.size = size
        self.canvas = canvas
        self.norm = norm

    def __call__(self, data, cache=False):
        d = copy.deepcopy(data)

        if cache:
            if "cache" in d.keys():
                img = np.float32(d["image"]) / 255.

                d["image"] = norm_image(img, self.norm)

                return d
            else:
                img = load_image(data['image_path'], self.size, self.canvas)

                img_storing = np.uint8((img * 255))

                d["image"] = norm_image(img, self.norm)

                return d, img_storing

        else: 
            img = load_image(data['image_path'], self.size, self.canvas)
            d["image"] = norm_image(img, self.norm)
            return d


def crop_im(img):

    t = threshold_li(img[0, :, :])
    binary = img[0, :, :] > t
    binary = getLargestCC(binary) 

    label_img = label(binary)
    regions = regionprops(label_img)
    areas = [r.area for r in regions]
    largest_cc_idx = np.argmax(areas)
    fov = regions[largest_cc_idx]

    cropped = img[:, fov.bbox[0]:fov.bbox[2], fov.bbox[1]: fov.bbox[3]]

    return cropped


def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC