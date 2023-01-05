from torchvision.models import swin_s
import torchvision
import torchvision.transforms as transforms
import agrocode.config as cfg
import torch
from tqdm import tqdm
import cv2
import numpy as np
import os
import pandas as pd


class ImageEmbeddings():
    def __init__(self, num_epochs: int, transformations: list, batch_size: int,
                 num_classes: int, use_saved_model: bool, model_args: dict = None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.use_saved_model = use_saved_model

        if use_saved_model:
            self.model = swin_s()
            self.model.head = torch.nn.Linear(self.model.head.in_features, num_classes, bias=False)
            self.model.load_state_dict(torch.load(cfg.WEIGHTS_PATH))
            self.model.to(self.device)
        else:
            self.model = swin_s(weights="DEFAULT")
            self.optimizer = torch.optim.Adam(self.model.parameters(), amsgrad=True, lr=0.0001)
            self.loss = torch.nn.CrossEntropyLoss()
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.01)

            self.num_epochs = num_epochs
            self.num_classes = num_classes
        print(f"Network will be trained on {self.device} device")
        # self.model = self.model.to(self.device)

        self.transformations = transformations
        self.batch_size = batch_size

        # print(self.model)
        # print(self.model.parameters())
        # print(self.model.avgpool)
        # for layer in self.model.parameters():
        #     print(layer)

    def prepare_datasets(self) -> None:
        img_transforms = []
        for trnsf in self.transformations:
            img_transforms.append(cfg.TRANSFORMS_DICT[trnsf])

        train_dataset_parts = []
        for trnsf in img_transforms:
            train_dataset_parts.append(
                torchvision.datasets.ImageFolder(
                    cfg.TRAIN_SORTED_IMAGES_PATH, transforms.Compose([transforms.Resize((224, 224)), trnsf] + cfg.BASE_TRANSFORMS)
                )
            )
        train_dataset_parts.append(
            torchvision.datasets.ImageFolder(
                cfg.TRAIN_SORTED_IMAGES_PATH, transforms.Compose([transforms.Resize((224, 224))] + cfg.BASE_TRANSFORMS)
            )
        )

        train_dataset = torch.utils.data.ConcatDataset(train_dataset_parts)
        val_dataset = torchvision.datasets.ImageFolder(
            cfg.VAL_IMAGES_PATH, transforms.Compose([transforms.Resize((224, 224))] + cfg.BASE_TRANSFORMS)
        )

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

        return None

    def prepare_model(self) -> None:
        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.model.head = torch.nn.Linear(self.model.head.in_features, self.num_classes, bias=False)
        self.model = self.model.to(self.device)

        return None
    
    def train_model(self) -> None:
 
        loss_hist = {'train':[], 'val':[]}
        acc_hist = {'train':[], 'val':[]}
    
        for epoch in range(self.num_epochs):
            print("\n Epoch {}/{}:".format(epoch, self.num_epochs - 1), end="")
            for phase in ['train', 'val']:
                if phase == 'train': #Если фаза == Тренировки  
                    dataloader = self.train_dataloader #берем train_dataLoader
                    #scheduler.step() #Делаем 1 шаг (произошла одна эпоха)
                    self.model.train()  # Модель в training mode - обучение (Фиксируем модель, иначе у нас могут изменяться параметры слоя батч-нормализации и изменится нейронка с течением времени)
                else: #Если фаза == Валидации 
                    dataloader = self.val_dataloader #берем val_dataLoader 
                    self.model.eval()   # Модель в evaluate mode - валидация (Фиксируем модель, иначе у нас могут изменяться параметры слоя батч-нормализации и изменится нейронка с течением времени)
    
                running_loss = 0. 
                running_acc = 0.
    
                # Итерируемся по dataloader
                for inputs, labels in tqdm(dataloader):
                    inputs = inputs.to(self.device) # Тензор с изображениями переводим на GPU 
                    labels = labels.to(self.device) # Тензор с лейблами переводим на GPU 
    
                    self.optimizer.zero_grad() # Обнуляем градиент,чтобы он не накапливался 
    
                    with torch.set_grad_enabled(phase == 'train'): #Если фаза train то активируем все градиенты (те которые не заморожены) (очистить историю loss)
                        preds = self.model(inputs) # Считаем предикты, input передаем в модель
                        loss_value = self.loss(preds, labels) #Посчитали  Loss    
                        preds_class = preds.argmax(dim=1) # Получаем класс,берем .argmax(dim=1) нейрон с максимальной активацией
                    
                        if phase == 'train':
                            loss_value.backward() # Считаем градиент 
                            self.optimizer.step() # Считаем шаг градиентного спуска
    
                    # Статистика
                    running_loss += loss_value.item() #считаем Loss
                    running_acc += (preds_class == labels.data).float().mean().data.cpu().numpy()  #считаем accuracy
    
                epoch_loss = running_loss / len(dataloader)  # Loss'ы делим на кол-во бачей в эпохе 
                epoch_acc = running_acc / len(dataloader) #считаем Loss на кол-во бачей в эпохе
    
                print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc), end="")
                
                loss_hist[phase].append(epoch_loss)
                acc_hist[phase].append(epoch_acc)
                if phase == 'train':
                    self.scheduler.step()
        torch.save(self.model.state_dict(), cfg.WEIGHTS_PATH)
        return None

    def make_embedding(self, path: str, csv_path: str = None) -> tuple:
        self.model.eval()

        image_indices = []
        image_embedding = []
        activation = {}

        def get_hidden_state(name):
            def hook(model, input, output):
                o = output.cpu().numpy()
                o_c = o.copy().reshape(-1).tolist()
                activation[name] = o_c  # .detach()
            return hook

        df = pd.read_csv(csv_path)
        for img_id in tqdm(df["idx"].tolist()):
        # for img_name in tqdm(os.listdir(path=path)):
        #     img_id = img_name.split(".")[0]
            img_name = f"{img_id}.png"
            img = cv2.imread(os.path.join(path, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            img = img / 255.0
            img = np.transpose(img, axes=(2, 0, 1))
            img = torch.from_numpy(img).float()
            img = torchvision.transforms.Resize((224, 224)).forward(img)
            img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).forward(img)
            img = img.unsqueeze_(0)
            img = img.to(self.device)

            with torch.no_grad():
                hs = self.model.avgpool.register_forward_hook(get_hidden_state("avgpool"))

                output = self.model(img)

            image_indices.append(img_id)
            image_embedding.append(activation["avgpool"])

            hs.remove()

        # print(image_embedding[0])
        # print(len(image_embedding[0]))
        return image_indices, image_embedding

    
    def run(self) -> tuple:
        if not self.use_saved_model:
            print("Preparing datasets")
            self.prepare_datasets()

            print("Preparing model")
            self.prepare_model()

            print("Trainig model")
            self.train_model()

        print("\nCreating queries images embeddings")
        queries_image_indices, queries_image_embeddings = self.make_embedding(cfg.QUERIES_IMAGES_PATH, cfg.QUERIES_TITLES_PATH)
        print("Creating test images embeddings")
        test_image_indices, test_image_embeddings = self.make_embedding(cfg.TEST_IMAGES_PATH, cfg.TEST_TITLES_PATH)

        return queries_image_indices, queries_image_embeddings, test_image_indices, test_image_embeddings
