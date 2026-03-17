import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import utils

# Avoid OpenCV oversubscription with DataLoader workers.
try:
    cv2.setNumThreads(0)
except Exception:
    pass


class AVLip(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val", "test"]
        self.data_label = opt.data_label

        self.real_list = utils.get_list(opt.real_list_path)
        self.fake_list = utils.get_list(opt.fake_list_path)

        self.label_dict = {p: 0 for p in self.real_list}
        self.label_dict.update({p: 1 for p in self.fake_list})

        self.total_list = self.real_list + self.fake_list
        if not self.total_list:
            raise ValueError(
                f"Empty dataset: real={len(self.real_list)} fake={len(self.fake_list)} split={self.data_label}"
            )

        self.targets = [self.label_dict[path] for path in self.total_list]
        self.class_paths = {
            0: self.real_list,
            1: self.fake_list,
        }

        # Build transforms once.
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        self.to_tensor = transforms.ToTensor()
        self.crop_idx = [(28, 196), (61, 163)]
        self.crop_resize = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.total_list)

    def _load_image_with_retry(self, img_path, label, max_retry=8):
        """Retry with same-class samples, then fail loudly.

        This prevents silently introducing zero images with non-zero labels.
        """
        class_candidates = self.class_paths.get(int(label), [])
        tried = set()
        current_path = img_path

        for _ in range(max_retry + 1):
            img_cv = cv2.imread(current_path)
            if img_cv is not None:
                return img_cv

            tried.add(current_path)
            if not class_candidates:
                break

            next_idx = int(np.random.randint(0, len(class_candidates)))
            current_path = class_candidates[next_idx]

            if current_path in tried and len(tried) < len(class_candidates):
                for candidate in class_candidates:
                    if candidate not in tried:
                        current_path = candidate
                        break

        raise FileNotFoundError(
            f"Failed to load image after retries (label={label}). Initial path: {img_path}"
        )

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.label_dict[img_path]

        # 1) Read BGR image
        img_cv = self._load_image_with_retry(img_path, label)

        # 2) BGR -> RGB
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        # 3) Build local crops
        crops_level0 = []
        for i in range(5):
            patch_cv = img_cv[500:, i * 500 : i * 500 + 500, :]
            if patch_cv.size == 0:
                patch_cv = np.zeros((500, 500, 3), dtype=np.uint8)

            patch_cv = cv2.resize(patch_cv, (224, 224), interpolation=cv2.INTER_LINEAR)
            patch_tensor = self.normalize(self.to_tensor(patch_cv))
            crops_level0.append(patch_tensor)

        crops = [crops_level0, [], []]

        for patch in crops_level0:
            c1 = patch[:, self.crop_idx[0][0] : self.crop_idx[0][1], self.crop_idx[0][0] : self.crop_idx[0][1]]
            c2 = patch[:, self.crop_idx[1][0] : self.crop_idx[1][1], self.crop_idx[1][0] : self.crop_idx[1][1]]
            crops[1].append(self.crop_resize(c1))
            crops[2].append(self.crop_resize(c2))

        # 4) Build global image tensor (normalization happens on GPU in trainer/validate/test)
        img_global = cv2.resize(img_cv, (1120, 1120), interpolation=cv2.INTER_LINEAR)
        img = self.to_tensor(img_global)

        return img, crops, label
