import argparse
import torch
from torchvision import transforms
from PIL import Image
import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg13()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
   
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image)
    img = img_transforms(img)
    return img.unsqueeze(0)  

def predict(image_path, model, top_k=1, category_names=None, device='cpu'):
    model.eval()
    image = process_image(image_path)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.exp(outputs)
        top_probs, top_indices = probabilities.topk(top_k)

    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name[str(idx.item())] for idx in top_indices[0]]
    else:
        top_classes = top_indices[0].tolist()

    return top_classes, top_probs[0].tolist()

def main():
    parser = argparse.ArgumentParser(description='Predict the class of an image.')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K classes')
    parser.add_argument('--category_names', type=str, default=None, help='Path to category to name mapping JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint)
    model.to(device)

    top_classes, top_probs = predict(args.image_path, model, args.top_k, args.category_names, device)

    print(f'Top classes: {top_classes}')
    print(f'Top probabilities: {top_probs}')

if __name__ == '__main__':
    main()
