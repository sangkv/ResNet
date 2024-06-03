import os
import time

import torch
from torchvision import transforms
from PIL import Image

class ImageClassifier:
    def __init__(self, model_dir):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_dir).to(self.device).eval()
        self.data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.softmax = torch.nn.Softmax(dim=1)
        self.class_names = ['ants', 'bees']
    
    def predict(self, img):
        img = self.data_transforms(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            outputs = self.softmax(outputs)
            _, preds = torch.max(outputs, 1)
            result = self.class_names[preds[0]]
        
        return result


classifier = ImageClassifier(model_dir='trained-models/fine-tuning-model.pth')

listFiles = os.listdir('data/test/')
sumTime = 0
for file in listFiles:
    image = Image.open('data/test/'+file)
    
    t0 = time.time()
    result = classifier.predict(image)
    t1 = time.time()
    sumTime = sumTime + (t1-t0)
    print("\nThe predicted result of image '%s' is: "%file, result)
print("\n\nSum time: ", sumTime, ", The average time: ", (sumTime/len(listFiles)))
