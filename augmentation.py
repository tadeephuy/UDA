from albumentations import Compose, RandomBrightness, ShiftScaleRotate


augmentations = [
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

augment = Compose(augmentations, p=1.0)

def apply_augmentations(img):
    data = {'image': img}
    augmented = augment(**data)['image']
    return augmented
