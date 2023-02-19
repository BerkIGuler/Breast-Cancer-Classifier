# site-packages
from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
from PIL import ImageFile

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

# local modules
from .utils import Plotter

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Model:

    def __init__(self, model_name, num_classes, feature_extract=0, use_pretrained=True):
        self.model_name = model_name
        self.feature_extract = feature_extract
        self.num_classes = num_classes
        self.initialize_model(model_name, num_classes, feature_extract, use_pretrained)

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained):
        self.model = None
        self.input_size = 0

        if model_name == "resnet":
            self.model = models.resnet50(pretrained=use_pretrained)
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "vgg":
            self.model = models.vgg16(pretrained=use_pretrained)
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,num_classes)
            self.input_size = 224

        elif model_name == "densenet":
            self.model = models.densenet161(pretrained=use_pretrained)
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, num_classes) 
            self.input_size = 224

        elif model_name == "inception":
            """ 
            Inception v3 expects (299,299) sized images and has auxiliary output
            """
            self.model = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(self.model, feature_extract)
            # Handle the auxilary net
            num_ftrs = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,num_classes)
            self.input_size = 299

        elif model_name == "b0":
            self.model = models.efficientnet_b0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = 1280
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes) 
            self.input_size = 224

        elif model_name == "b2":
            self.model = models.efficientnet_b2(pretrained=use_pretrained)
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes) 
            self.input_size = 260

        elif model_name == "mobileNet":
            self.model = models.mobilenet_v3_large(pretrained=True)
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = 1280
            self.model.classifier[3] = nn.Linear(num_ftrs, num_classes) 
            self.input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        print("Initializing Datasets and Dataloaders...")
        print(self.model)

    @staticmethod
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting == 1:
            for param in model.parameters():
                param.requires_grad = False

    def get_params_to_update(self):
        params_to_update = self.model.parameters()
        if self.feature_extract == 1:
            params_to_update = []
            for name,param in self.model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)

        return params_to_update


class TrainingArguments():
    def __init__(self, batch_size, num_epochs, criterion, model_instance, gpu_id, early_stop_patience, lr, evaluate_every_n_iter):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.gpu_id = gpu_id
        self.optimizer = optim.Adam(model_instance.get_params_to_update(), lr, betas=(0.9, 0.999), eps=1e-08)
        self.patience = early_stop_patience
        self.evaluate_every_n_iter = evaluate_every_n_iter


class Dataset():
    def __init__(self, dataset_name, training_args_instance, model_instance, apply_augmentation=0, num_workers=4):
        self.training_args = training_args_instance
        self.apply_augmentation = apply_augmentation
        self.model = model_instance
        self.dataset_name = dataset_name
        self.num_workers = num_workers
        self.set_data_transforms(apply_augmentation)
        self.set_dataset_path(dataset_name)
        self.set_data_loaders()

    def set_data_transforms(self, apply_augmentation):
        if apply_augmentation == 1:
            self.data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(self.model.input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply(torch.nn.ModuleList([
                                               transforms.RandomRotation(30)]), p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test': transforms.Compose([
                    transforms.Resize(self.model.input_size),
                    transforms.CenterCrop(self.model.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }
        elif apply_augmentation == 0:
            self.data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(self.model.input_size),
                    transforms.CenterCrop(self.model.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test': transforms.Compose([
                    transforms.Resize(self.model.input_size),
                    transforms.CenterCrop(self.model.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            }
        else:
            raise "Invalid input for --augment"


    def set_dataset_path(self, dataset_name):

        if dataset_name == "ISIC":
            self.data_path = "/auto/data2/bguler/DDAN/ISIC"
        elif dataset_name == "retina":
            self.data_path = "/auto/data2/bguler/DDAN/retina"
        elif dataset_name == "hyper":
            self.data_path = "/auto/data2/bguler/DDAN/hyper_kvasir"

        elif dataset_name == "aug_hyper":
            self.data_path = "/auto/data2/bguler/DDAN/aug_hyper_kvasir"
        elif dataset_name == "aug_retina":
            self.data_path = "/auto/data2/bguler/DDAN/aug_retina"
        elif dataset_name == "aug_ISIC":
            self.data_path = "/auto/data2/bguler/DDAN/aug_ISIC"

        elif dataset_name == "semi_aug_100_hyper":
            self.data_path = "/auto/data2/bguler/DDAN/semi_aug_100_hyper_kvasir"
        elif dataset_name == "semi_aug_100_retina":
            self.data_path = "/auto/data2/bguler/DDAN/semi_aug_100_retina"
        elif dataset_name == "semi_aug_100_ISIC":
            self.data_path = "/auto/data2/bguler/DDAN/semi_aug_100_ISIC"

        elif dataset_name == "semi_aug_25_hyper":
            self.data_path = "/auto/data2/bguler/DDAN/semi_aug_25_hyper_kvasir"
        elif dataset_name == "semi_aug_25_retina":
            self.data_path = "/auto/data2/bguler/DDAN/semi_aug_25_retina"
        elif dataset_name == "semi_aug_25_ISIC":
            self.data_path = "/auto/data2/bguler/DDAN/semi_aug_25_ISIC"


        elif dataset_name == "retina_25_ddpm":
            self.data_path = "/auto/data2/bguler/DDAN/retina_25_ddpm"
        elif dataset_name == "ISIC_25_ddpm":
            self.data_path = "/auto/data2/bguler/DDAN/ISIC_25_ddpm"

        elif dataset_name == "retina_100_ddpm":
            self.data_path = "/auto/data2/bguler/DDAN/retina_100_ddpm"
        elif dataset_name == "ISIC_100_ddpm":
            self.data_path = "/auto/data2/bguler/DDAN/ISIC_100_ddpm"        

        elif dataset_name == "retina_200_ddpm":
            self.data_path = "/auto/data2/bguler/DDAN/retina_200_ddpm"
        elif dataset_name == "ISIC_200_ddpm":
            self.data_path = "/auto/data2/bguler/DDAN/ISIC_200_ddpm"      

        elif dataset_name == "retina_300_ddpm":
            self.data_path = "/auto/data2/bguler/DDAN/retina_300_ddpm"
        elif dataset_name == "ISIC_300_ddpm":
            self.data_path = "/auto/data2/bguler/DDAN/ISIC_300_ddpm"     

        else:
            print("invalid dataset. Exiting...")
            exit()


    def set_data_loaders(self):
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_path, x), self.data_transforms[x]) for x in ['train', 'test']}
        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.num_workers) \
                                                                                                                           for x in ['train', 'test']}
    def get_test_dataloader(self):
        # Create test dataset and dataloader
        test_dataset = datasets.ImageFolder(os.path.join(self.data_path, "test"), self.data_transforms["test"])
        dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=self.num_workers)

        return dataloader_test


class Trainer():
    def __init__(self, training_args_instance, dataset_instance, model_instance):

        self.device = torch.device(f"cuda:{training_args_instance.gpu_id}" if torch.cuda.is_available() else "cpu")
        print("training device set as:", self.device)
        self.training_args = training_args_instance
        self.dataset = dataset_instance
        model_instance.model = model_instance.model.to(self.device)
        self.model = model_instance


    def train(self):
        since = time.time()

        val_acc_history = []
        train_acc_history = []
        train_loss_history = []
        val_loss_history = []
        
        best_model_wts = copy.deepcopy(self.model.model.state_dict())
        best_loss = 100000
        remaining_patience = self.training_args.patience
        quit = False

        for epoch in range(self.training_args.num_epochs):

            if quit:
                break

            print('Epoch {}/{}'.format(epoch, self.training_args.num_epochs - 1))
            print('-' * 100)

            running_loss_train = 0.0
            running_corrects_train = 0

            num_iter_train = 0

            for inputs, labels in self.dataset.dataloaders["train"]:

                self.model.model.train()

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.training_args.optimizer.zero_grad()


                with torch.set_grad_enabled(True):
                    if self.model.model_name == "inception":
                        outputs, aux_outputs = self.model.model(inputs)
                        loss1 = self.training_args.criterion(outputs, labels)
                        loss2 = self.training_args.criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = self.model.model(inputs)
                        loss = self.training_args.criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    loss.backward()
                    self.training_args.optimizer.step()

                running_loss_train += loss.item() * inputs.size(0)
                running_corrects_train += torch.sum(preds == labels.data)
                num_iter_train += 1


                if num_iter_train % self.training_args.evaluate_every_n_iter == 0:

                    self.model.model.eval()

                    if remaining_patience <= 0:
                        quit = True
                        break

                    running_loss_test = 0.0
                    running_corrects_test = 0

                    for inputs, labels in self.dataset.dataloaders["test"]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        with torch.no_grad():
                            if self.model.model_name == "inception":

                                outputs = self.model.model(inputs)
                                loss = self.training_args.criterion(outputs, labels)
                            else:
                                outputs = self.model.model(inputs)
                                loss = self.training_args.criterion(outputs, labels)

                            _, preds = torch.max(outputs, 1)

                        running_loss_test += loss.item() * inputs.size(0)
                        running_corrects_test += torch.sum(preds == labels.data)


                    iter_loss_test = running_loss_test / (len(self.dataset.dataloaders["test"].dataset))
                    iter_acc_test = running_corrects_test.double() / (len(self.dataset.dataloaders["test"].dataset))
                    iter_loss_train = running_loss_train / (num_iter_train * self.training_args.batch_size)
                    iter_acc_train = running_corrects_train / (num_iter_train * self.training_args.batch_size)


                    if iter_loss_test < best_loss:
                        best_loss = iter_loss_test
                        best_loss_acc =  iter_acc_test
                        best_model_wts = copy.deepcopy(self.model.model.state_dict())
                        remaining_patience = self.training_args.patience

                        

                    else:
                        remaining_patience -= 1

                    
                    val_acc_history.append(iter_acc_test)
                    val_loss_history.append(iter_loss_test)

                    train_acc_history.append(iter_acc_train)
                    train_loss_history.append(iter_loss_train)


                    print(f"EPOCH {epoch}/{self.training_args.num_epochs - 1}, ITER {num_iter_train}, DATASET {self.dataset.dataset_name}")
                    print(f"Training Set\t\t Accuracy: {iter_acc_train:.4f}\t\t Loss: {iter_loss_train:.4f}")
                    print(f"Test Set\t\t Accuracy: {iter_acc_test:.4f}\t\t Loss: {iter_loss_test:.4f}")
                    print("remaining_patience:", remaining_patience)


            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best Test Acc: {:4f}'.format(best_loss_acc))
        self.model.model.load_state_dict(best_model_wts)

        database = {
            'train loss': train_loss_history,
            'val loss': val_loss_history,
            'train acc': train_acc_history,
            'val acc': val_acc_history}

        self.best_acc = best_loss_acc
        return database


    def save_checkpoints(self, database, test_dataloader):

        if self.dataset.apply_augmentation == 1:
            tag = f"basic_aug_{self.model.model_name}_{self.dataset.dataset_name}_{self.best_acc:2f}"
        elif self.dataset.apply_augmentation == 0:
            tag = f"{self.model.model_name}_{self.dataset.dataset_name}_{self.best_acc:2f}"  
        else:
            raise "invalid --augment flag"

        current_dir = os.getcwd()
        records_dir = os.path.join(current_dir, "records")
        checkpoints_dir = os.path.join(records_dir, tag)
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)

        ims_dir = os.path.join(checkpoints_dir, tag + "_ims")

        if not os.path.exists(ims_dir):
            os.mkdir(ims_dir)

        # save weights on disk
        torch.save(self.model.model.state_dict(), checkpoints_dir + "/weights.pth")



        # save metrics
        metrics_path = os.path.join(checkpoints_dir, "metrics.txt")
        labs, outs = self.predict(test_dataloader)
        self.save_metrics(labs, outs, metrics_path)


        Plotter.plot_simple_acc(database['train acc'], ims_dir + "/train_acc.png")
        Plotter.plot_simple_acc(database['val acc'], ims_dir + "/test_acc.png")
        Plotter.plot_simple_loss(database['train loss'], ims_dir + "/train_loss.png")
        Plotter.plot_simple_loss(database['val loss'], ims_dir + "/test_loss.png")

        Plotter.plot_double_acc(database['train acc'], database['val acc'], ims_dir + "/double_acc.png")
        Plotter.plot_double_loss(database['train loss'], database['val loss'], ims_dir + "/double_loss.png")

        print("saved checkpoints to {}".format(checkpoints_dir))


    def predict(self, dataloader):
        since = time.time()
        labs = torch.zeros(len(dataloader.dataset), self.model.num_classes)
        outs = torch.zeros(len(dataloader.dataset), self.model.num_classes)

        self.model.model.eval()   # Set model to evaluate mode

        running_corrects = 0
        batch_ind = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            labels_one_hot = nn.functional.one_hot(labels, num_classes=self.model.num_classes)

            with torch.no_grad():
                outputs = self.model.model(inputs)
                outs[batch_ind, :] = outputs
                labs[batch_ind, :] = labels_one_hot
                _, preds_temp = torch.max(outputs, 1)
                # statistics
                running_corrects += torch.sum(preds_temp == labels.data)
            batch_ind += 1

        acc = running_corrects.double() / len(dataloader.dataset)



        print(f"Test set accuracy: {acc:4f}")
        time_elapsed = time.time() - since

        return labs, outs

    def save_metrics(self, labs, outs, save_path):

        _, preds = torch.max(outs, 1)
        y_preds_one_hot = nn.functional.one_hot(preds, num_classes = self.model.num_classes)


        # one hot labels
        labels = labs.numpy()

        # output probabilities
        outputs = outs.numpy()

        # one hot predictions
        y_preds_one_hot = y_preds_one_hot.numpy()


        # convert to int
        labels = labels.astype(int)
        y_preds_one_hot = y_preds_one_hot.astype(int)


        single_labels=np.argmax(labels, axis=1)
        single_preds=np.argmax(y_preds_one_hot, axis=1)


        self.save_stats(single_labels, single_preds, save_path)



    def save_stats(self, y_test, y_pred, save_path):

        class_list = []
        for i in range(self.model.num_classes):
            class_list.append("c-{}".format(i))


        cm_path = os.path.join(os.path.dirname(save_path), "cm.png")
        Plotter.plot_cm(y_test, y_pred, class_list, cm_path)


        print('\nAccuracy: {:.4f}\n'.format(accuracy_score(y_test, y_pred)))

        print('Micro Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='micro')))
        print('Micro Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='micro')))
        print('Micro F1-score: {:.4f}\n'.format(f1_score(y_test, y_pred, average='micro')))

        print('Macro Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='macro')))
        print('Macro Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='macro')))
        print('Macro F1-score: {:.4f}\n'.format(f1_score(y_test, y_pred, average='macro')))

        print('Weighted Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='weighted')))
        print('Weighted Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='weighted')))
        print('Weighted F1-score: {:.4f}'.format(f1_score(y_test, y_pred, average='weighted')))


        print('\nClassification Report\n')
        print(classification_report(y_test, y_pred, target_names=class_list, digits=4))

        with open(save_path, "w") as f_out:

            f_out.write('\nAccuracy: {:.4f}\n'.format(accuracy_score(y_test, y_pred)))

            f_out.write('Micro Precision: {:.4f}\t'.format(precision_score(y_test, y_pred, average='micro')))
            f_out.write('Micro Recall: {:.4f}\t'.format(recall_score(y_test, y_pred, average='micro')))
            f_out.write('Micro F1-score: {:.4f}\n'.format(f1_score(y_test, y_pred, average='micro')))

            f_out.write('Macro Precision: {:.4f}\t'.format(precision_score(y_test, y_pred, average='macro')))
            f_out.write('Macro Recall: {:.4f}\t'.format(recall_score(y_test, y_pred, average='macro')))
            f_out.write('Macro F1-score: {:.4f}\n'.format(f1_score(y_test, y_pred, average='macro')))

            f_out.write('Weighted Precision: {:.4f}\t'.format(precision_score(y_test, y_pred, average='weighted')))
            f_out.write('Weighted Recall: {:.4f}\t'.format(recall_score(y_test, y_pred, average='weighted')))
            f_out.write('Weighted F1-score: {:.4f}\n'.format(f1_score(y_test, y_pred, average='weighted')))


            f_out.write('\nClassification Report\n')
            f_out.write(classification_report(y_test, y_pred, target_names=class_list, digits=4))
            