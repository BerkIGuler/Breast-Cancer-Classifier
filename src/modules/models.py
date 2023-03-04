from modules.utils import Plotter, Statistics
import modules.logger as logger

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_LOSS = 1000000


class Model:

    def __init__(self, args, use_pretrained=True):
        self.model_name = args.model
        self.feature_extract = args.feature_extract
        self.num_classes = args.num_classes
        self.logger = logger.get_global_logger(args, name=__name__)
        self.model, self.input_size = self.initialize_model(
            args.model, args.num_classes,
            args.feature_extract, use_pretrained)

    def initialize_model(
            self, model_name, num_classes, feature_extract, use_pretrained
    ):
        if model_name == "resnet50":
            model = models.resnet50(pretrained=use_pretrained)
            self._set_parameter_requires_grad(model, feature_extract)
            num_ftrs = self.model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg16":
            model = models.vgg16(pretrained=use_pretrained)
            self._set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "densenet161":
            model = models.densenet161(pretrained=use_pretrained)
            self._set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            # Inception v3 expects (299,299) sized images and has auxiliary output
            model = models.inception_v3(pretrained=use_pretrained)
            self._set_parameter_requires_grad(model, feature_extract)
            # Handle the auxilary net
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299

        elif model_name == "b0":
            model = models.efficientnet_b0(pretrained=use_pretrained)
            self._set_parameter_requires_grad(model, feature_extract)
            num_ftrs = 1280
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "b2":
            model = models.efficientnet_b2(pretrained=use_pretrained)
            self._set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            input_size = 260

        elif model_name == "mobileNetv3Large":
            model = models.mobilenet_v3_large(pretrained=True)
            self._set_parameter_requires_grad(model, feature_extract)
            num_ftrs = 1280
            model.classifier[3] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        else:
            raise ValueError("invalid model name")

        self.logger.info(f"Initialized the model {model_name}")
        return model, input_size

    @staticmethod
    def _set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting == 1:
            for param in model.parameters():
                param.requires_grad = False

    def get_params_to_update(self):
        params_to_update = self.model.parameters()
        if self.feature_extract == 1:
            params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)

        return params_to_update


class TrainingArguments:
    def __init__(self, args, criterion, model_instance):
        self.batch_size = args.batch_size
        self.num_steps = args.num_steps
        self.criterion = criterion
        self.gpu_id = args.gpu_id
        self.optimizer = optim.Adam(
            model_instance.get_params_to_update(),
            args.lr, betas=(0.9, 0.999), eps=1e-08)
        self.patience = args.patience
        self.evaluate_every_n_iter = args.eval_freq


class Dataset:
    def __init__(self, args, training_args_instance, model_instance):
        self.training_args = training_args_instance
        self.apply_augmentation = args.augment
        self.model = model_instance
        self.dataset_name = args.dataset
        self.num_workers = args.num_workers
        self.data_transforms = self._get_data_transforms(args.augment)
        self.dataset_path = self._get_dataset_path(args.dataset)
        self.dataloaders = self._get_data_loaders()
        self.logger = logger.get_global_logger(args, name=__name__)

    def _get_data_transforms(self, apply_augmentation):
        if apply_augmentation == 1:
            data_transforms = {
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
            data_transforms = {
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
            raise ValueError("Invalid input for --augment")
        return data_transforms

    @staticmethod
    def _get_dataset_path(dataset_name):
        if dataset_name == "breast_cancer":
            data_path = "/auto/data2/bguler/DDAN/breast_cancer"
        else:
            raise ValueError("invalid dataset.")
        return data_path

    def _get_data_loaders(self):
        image_datasets = {x: datasets.ImageFolder(
            os.path.join(self.dataset_path, x),
            self.data_transforms[x]) for x in ['train', 'test']}
        dataloaders = {x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=self.training_args.batch_size,
            shuffle=True, num_workers=self.num_workers) for x in ['train', 'test']}
        return dataloaders

    def get_test_dataloader(self):
        # Create test dataset and dataloader
        test_dataset = datasets.ImageFolder(
            os.path.join(self.dataset_path, "test"),
            self.data_transforms["test"])
        dataloader_test = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=True,
            num_workers=self.num_workers)

        return dataloader_test


class Trainer:
    def __init__(
            self, args, training_args_instance,
            dataset_instance, model_instance):

        self.device = torch.device(f"cuda:{training_args_instance.gpu_id}" if
                                   torch.cuda.is_available() else "cpu")

        self.training_args = training_args_instance
        self.dataset = dataset_instance
        model_instance.model = model_instance.model.to(self.device)
        self.model = model_instance
        self.logger = logger.get_global_logger(args, name=__name__)
        self.logger.info(f"training device set as: {self.device}")

    def train(self):
        since = time.time()

        stats = Statistics(columns=['train loss', 'val loss', 'train acc', 'val acc'])
        
        best_model_wts = copy.deepcopy(self.model.model.state_dict())
        best_loss = MAX_LOSS
        best_loss_acc = 0
        remaining_patience = self.training_args.patience
        quit_flag = False

        running_loss_train = 0.0
        running_corrects_train = 0
        num_iter_train = 0

        while True:
            if quit_flag:
                break

            for inputs, labels in self.dataset.dataloaders["train"]:

                if num_iter_train >= self.training_args.num_steps:
                    quit_flag = True
                    break

                self.model.model.train()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                with torch.set_grad_enabled(True):
                    self.training_args.optimizer.zero_grad()
                    if self.model.model_name == "inception":
                        outputs, aux_outputs = self.model.model(inputs)
                        loss1 = self.training_args.criterion(outputs, labels)
                        loss2 = self.training_args.criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = self.model.model(inputs)
                        loss = self.training_args.criterion(outputs, labels)
                    loss.backward()
                    self.training_args.optimizer.step()

                _, preds = torch.max(outputs, dim=1)
                running_loss_train += loss.item() * inputs.size(0)
                running_corrects_train += torch.sum(preds == labels.data)
                num_iter_train += 1
                if num_iter_train % self.training_args.evaluate_every_n_iter == 0:
                    self.model.model.eval()
                    if remaining_patience <= 0:
                        quit_flag = True
                        break
                    running_loss_test = 0.0
                    running_corrects_test = 0
                    for test_inputs, test_labels in self.dataset.dataloaders["test"]:
                        test_inputs = test_inputs.to(self.device)
                        test_labels = test_labels.to(self.device)

                        with torch.no_grad():
                            outputs = self.model.model(test_inputs)
                            loss = self.training_args.criterion(outputs, test_labels)
                            _, preds = torch.max(outputs, 1)

                        running_loss_test += loss.item() * test_inputs.size(0)
                        running_corrects_test += torch.sum(preds == test_labels.data)

                    iter_loss_test = running_loss_test \
                        / (len(self.dataset.dataloaders["test"].dataset))
                    iter_acc_test = running_corrects_test \
                        / (len(self.dataset.dataloaders["test"].dataset))
                    iter_loss_train = running_loss_train \
                        / (self.training_args.evaluate_every_n_iter * self.training_args.batch_size)
                    iter_acc_train = running_corrects_train \
                        / (self.training_args.evaluate_every_n_iter * self.training_args.batch_size)

                    running_loss_train = 0.0
                    running_corrects_train = 0

                    if iter_loss_test < best_loss:
                        best_loss = iter_loss_test
                        best_loss_acc = iter_acc_test
                        best_model_wts = copy.deepcopy(self.model.model.state_dict())
                        remaining_patience = self.training_args.patience
                    else:
                        remaining_patience -= 1

                    stats.add_single_data_entry(
                        col_names=['train loss', 'val loss', 'train acc', 'val acc'],
                        col_data=[iter_loss_train,
                                  iter_loss_test,
                                  iter_acc_train.cpu().numpy(),
                                  iter_acc_test.cpu().numpy()
                        ]
                    )

                    print(f"ITER {num_iter_train}, DATASET {self.dataset.dataset_name}")
                    print(f"Training Set\t\t Accuracy: {iter_acc_train:.4f}"
                          + f"\t\t Loss: {iter_loss_train:.4f}")
                    print(f"Test Set\t\t Accuracy: {iter_acc_test:.4f}\t\t"
                          + f" Loss: {iter_loss_test:.4f}")
                    print("remaining_patience:", remaining_patience)

        time_elapsed = time.time() - since
        self.logger.info(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        self.logger.info(f'Best Test Acc: {best_loss_acc:.4f}')
        self.model.model.load_state_dict(best_model_wts)

        return stats.get_statistics(), best_loss_acc

    def save_checkpoints(self, database, best_acc, test_dataloader):
        if self.dataset.apply_augmentation == 1:
            tag = f"aug_{self.model.model_name}_{self.dataset.dataset_name}_{best_acc:2f}"
        elif self.dataset.apply_augmentation == 0:
            tag = f"{self.model.model_name}_{self.dataset.dataset_name}_{best_acc:2f}"
        else:
            raise ValueError("invalid --augment flag")

        parent_dir = os.path.dirname(os.getcwd())
        records_dir = os.path.join(parent_dir, "results")
        checkpoints_dir = os.path.join(records_dir, tag)
        metrics_path = os.path.join(checkpoints_dir, "metrics.txt")
        os.makedirs(checkpoints_dir, exist_ok=True)
        ims_dir = os.path.join(checkpoints_dir, tag + "_ims")
        os.makedirs(ims_dir, exist_ok=True)

        # save weights on disk
        torch.save(self.model.model.state_dict(), checkpoints_dir + "/weights.pth")

        # save metrics
        labs, outs = self.predict(test_dataloader)
        self.save_metrics(labs, outs, metrics_path)

        Plotter.plot_simple_acc(database['train acc'], ims_dir + "/train_acc.png")
        Plotter.plot_simple_acc(database['val acc'], ims_dir + "/test_acc.png")
        Plotter.plot_simple_loss(database['train loss'], ims_dir + "/train_loss.png")
        Plotter.plot_simple_loss(database['val loss'], ims_dir + "/test_loss.png")

        Plotter.plot_double_acc(database['train acc'], database['val acc'], ims_dir + "/double_acc.png")
        Plotter.plot_double_loss(database['train loss'], database['val loss'], ims_dir + "/double_loss.png")

        self.logger.info(f"saved checkpoints to {checkpoints_dir}")

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
            labels_one_hot = nn.functional.one_hot(
                labels, num_classes=self.model.num_classes
            )

            with torch.no_grad():
                outputs = self.model.model(inputs)
                outs[batch_ind, :] = outputs
                labs[batch_ind, :] = labels_one_hot
                _, preds_temp = torch.max(outputs, 1)
                running_corrects += torch.sum(preds_temp == labels.data)
            batch_ind += 1

        acc = running_corrects / len(dataloader.dataset)

        self.logger.info(f"Test set accuracy: {acc:4f}")
        time_elapsed = time.time() - since
        self.logger.info(f"Inference on test set took: {time_elapsed:4f}")
        return labs, outs

    def save_metrics(self, labs, outs, save_path):
        _, preds = torch.max(outs, 1)
        y_preds_one_hot = nn.functional.one_hot(preds, num_classes=self.model.num_classes)

        # one hot labels
        labels = labs.numpy()
        # one hot predictions
        y_preds_one_hot = y_preds_one_hot.numpy()

        labels = labels.astype(int)
        y_preds_one_hot = y_preds_one_hot.astype(int)

        single_labels = np.argmax(labels, axis=1)
        single_preds = np.argmax(y_preds_one_hot, axis=1)

        self.save_stats(single_labels, single_preds, save_path)

    def save_stats(self, y_test, y_pred, save_path):
        class_list = []
        for i in range(self.model.num_classes):
            class_list.append(f"c-{i}")

        cm_path = os.path.join(os.path.dirname(save_path), "cm.png")
        Plotter.plot_cm(y_test, y_pred, class_list, cm_path)

        with open(save_path, "w") as f_out:

            f_out.write(f'\nAccuracy:\n {accuracy_score(y_test, y_pred):.4f}')
            f_out.write("Micro Precision: "
                        + f"{precision_score(y_test, y_pred, average='micro'):.4f}\t")
            f_out.write("Micro Recall: "
                        + f"{recall_score(y_test, y_pred, average='micro'):.4f}\t")
            f_out.write("Micro F1-score: "
                        + f"{f1_score(y_test, y_pred, average='micro'):.4f}\n")
            f_out.write("Macro Precision: "
                        + f"{precision_score(y_test, y_pred, average='macro'):.4f}\t")
            f_out.write("Macro Recall: "
                        + f"{recall_score(y_test, y_pred, average='macro'):.4f}\t")
            f_out.write("Macro F1-score: "
                        + f"{f1_score(y_test, y_pred, average='macro'):.4f}\n")
            f_out.write("Weighted Precision: "
                        + f"{precision_score(y_test, y_pred, average='weighted'):.4f}\t")
            f_out.write("Weighted Recall: "
                        + f"{recall_score(y_test, y_pred, average='weighted'):.4f}\t")
            f_out.write("Weighted F1-score: "
                        + f"{f1_score(y_test, y_pred, average='weighted'):.4f}\n")
            f_out.write('\nClassification Report\n')
            f_out.write(classification_report(
                y_test, y_pred, target_names=class_list,
                digits=4)
            )
            