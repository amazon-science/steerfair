from torchvision.transforms import v2


def random_perturb_image(img):
    transform = v2.Compose(
        [
            v2.PILToTensor(),
            # v2.RandomHorizontalFlip(0.5),
            # v2.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            v2.RandomZoomOut(side_range=(1.0, 2.0)),
            v2.RandomGrayscale(0.5),
            v2.RandomRotation(45),
            v2.ToPILImage(),
        ]
    )
    img = transform(img)
    return img
