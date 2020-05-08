from torchvision.transforms import Compose, Resize, Normalize, ToTensor

def get_transforms(img_size):
    return Compose([
        Resize([img_size, img_size]),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])