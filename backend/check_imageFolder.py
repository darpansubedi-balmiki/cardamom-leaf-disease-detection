from torchvision import datasets

ds = datasets.ImageFolder("./dataset/train")
print("classes:", ds.classes)
print("class_to_idx:", ds.class_to_idx)