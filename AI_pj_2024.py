from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torchsummary import summary
import argparse
import numpy as np
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.metrics import f1_score
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms

# 모델 학습 관련 파라미터 모음 --> 자유롭게 변경하고, 추가해보세요.

class Args():
  data_type = "2d"
  scheduler = "multistep"
  model = "resnet"
  n_class = 3
  epoch = 5
  phase = "train"
  model_path = "./model_weight_2d.pth"

args = Args()

class TimeMasking(object):
    def __init__(self, T=40, max_masks=1):
        self.T = T
        self.max_masks = max_masks

    def __call__(self, spec):
        for _ in range(0, self.max_masks):
            t = random.randrange(0, self.T)
            t0 = random.randrange(0, spec.shape[1] - t)
            spec[:, t0:t0+t] = 0
        return spec

class FrequencyMasking(object):
    def __init__(self, F=30, max_masks=1):
        self.F = F
        self.max_masks = max_masks

    def __call__(self, spec):
        for _ in range(0, self.max_masks):
            f = random.randrange(0, self.F)
            max_f0 = spec.shape[0] - f
            if max_f0 <= 0:
                continue
            f0 = random.randrange(0, max_f0)
            spec[f0:f0+f, :] = 0
        return spec

class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.classes = sorted(os.listdir(directory))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []


        for class_name in self.classes:
           class_dir = os.path.join(directory, class_name)
           for image_name in os.listdir(class_dir):
               self.samples.append((os.path.join(class_dir, image_name), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def get_model(name, n_class, pretrained=True):
    if name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, n_class)
    elif name == 'resnet':
        model = models.resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, n_class)

class Simple2DCNN(nn.Module):
    def __init__(self):
        super(Simple2DCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((10, 10))
        self.fc = nn.Linear(256 * 7 * 7, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc(x)
        return x

def visualize_audio_batch(audio_signals, labels):
    fig, axes = plt.subplots(4, 4, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        if i >= 16:
            break
        ax.plot(audio_signals[i].t().numpy())
        ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def tic():
    global start_time
    start_time = time.time()

def toc():
    elapsed_time = time.time() - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print(f"학습에 소요된 시간은 총 : {hours}시간 {minutes}분 {seconds}초 입니다.")

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, true_labels = [], []
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for imgs, labels in iter(val_loader):
            imgs = imgs.float().to(device)
            labels = labels.long().to(device)

            pred = model(imgs)

            loss = criterion(pred, labels)

            preds += pred.argmax(1).detach().cpu().numpy().tolist()

            _, predicted = torch.max(pred, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            true_labels += labels.detach().cpu().numpy().tolist()

            val_loss.append(loss.item())

        _val_accuracy = 100 * correct_predictions / total_predictions

        _val_loss = np.mean(val_loss)
        average = 'macro'
        _val_score = f1_score(true_labels, preds, average='macro')

    return _val_loss, _val_accuracy

def train_model(model, train_loader, epochs, device, args):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if args.scheduler == 'multistep':
        scheduler = MultiStepLR(optimizer, [5,10], gamma=0.1)
    elif args.scheduler == 'steplr':
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        correct_predictions = 0
        total_samples = 0

        for i, (audio_signals, labels) in enumerate(tqdm(train_loader)):
            audio_signals, labels = audio_signals.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(audio_signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = running_loss / total_batches
        accuracy = correct_predictions / total_samples
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        scheduler.step()

    print('Finished Training')

    torch.save(model.state_dict(), f"./model_weight_{args.data_type}.pth")
    return model

if args.data_type == "1d":

    model = Simple1DCNN()


    train_dataset = AudioDataset(directory='dataset/train')
    val_dataset = AudioDataset(directory='dataset/train')

elif args.data_type == "2d":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        TimeMasking(T=40, max_masks=1),
        FrequencyMasking(F=30, max_masks=1)
    ])

    if args.model == 'vgg16':
        model = get_model(args.model, args.n_class, pretrained=True)
    elif args.model == 'resnet':
        model = get_model(args.model, args.n_class, pretrained=True)
    elif args.model == 'simple':
        model = Simple2DCNN()

    train_dataset = ImageDataset(directory='dataset/train', transform=transform)
    #val_dataset = ImageDataset(directory='dataset/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(device)
print(f"학습 데이터 수는 {len(train_dataset)}개 입니다.")
#print(f"검증 데이터 수는 {len(val_dataset)}개 입니다.")

if args.data_type == "1d":
    summary(model.cuda(), (1,16000))
elif args.data_type == "2d":
    summary(model.cuda(), (3,224,224))

tic()

model = train_model(model, train_loader, epochs=args.epoch, device=device, args=args)

toc()

 if args.data_type == "1d":

    test_dataset = AudioDataset(directory='dataset/test')

elif args.data_type == "2d":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset = ImageDataset(directory='dataset/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"테스트 데이터 수는 {len(test_dataset)}개 입니다.")

def evaluate_model(model, test_loader, device, args):
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total * 100
    f1 = f1_score(all_labels, all_predictions, average='macro')
    print(f'Test Accuracy: {accuracy:.2f}%, Avg Loss: {avg_loss:.4f}, F1 Score: {f1:.4f}')

    return accuracy, avg_loss, f1

accuracy, avg_loss, f1 = evaluate_model(model, test_loader, device=device, args=args)

print(f"테스트 데이터의 f1 score는 {f1}")
