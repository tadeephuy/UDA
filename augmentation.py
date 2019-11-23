from albumentations import Compose, RandomBrightness, ShiftScaleRotate


augmentations_sup = [
    RandomBrightness(limit=0.067,
                p=0.112256),
    ShiftScaleRotate(p=0.308451,
                shift_limit=0,
                rotate_limit=5.001245,
                scale_limit=0),
    ShiftScaleRotate(p=0.135129,
                shift_limit=0.109086,
                rotate_limit=0,
                scale_limit=0),
    ShiftScaleRotate(p=0.094869,
                shift_limit=0,
                rotate_limit=0,
                scale_limit=0.150789)
]
augmentations_unsup = [
    RandomBrightness(limit=0.2,
                p=0.112256),
    ShiftScaleRotate(p=0.308451,
                shift_limit=0,
                rotate_limit=15.001245,
                scale_limit=0),
    ShiftScaleRotate(p=0.135129,
                shift_limit=1.109086,
                rotate_limit=0,
                scale_limit=0),
    ShiftScaleRotate(p=0.2,
                shift_limit=0,
                rotate_limit=0,
                scale_limit=0.150789)
]


def apply_augmentations_sup(img):
    data = {'image': img}
    augmented = Compose(augmentations_sup, p=0.5)(**data)['image']
    return augmented

def apply_augmentations_unsup(img):
    data = {'image': img}
    augmented = Compose(augmentations_unsup, p=1.0)(**data)['image']
    return augmented

def apply_augmentations_tta(img):
    data = {'image': img}
    augmented = Compose(augmentations_sup, p=1)(**data)['image']
    return augmented