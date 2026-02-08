import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import dataset_utils 
import training_loop
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def train_classifier(classifier, 
                     optimizer, 
                     criterion, 
                     data_loader, 
                     num_epochs, 
                     device, 
                     checkpoint_path, 
                     model_name,
                     low_res_input=True):
    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (labels, low_res, high_res) in enumerate(data_loader):
            #images, labels = images.to(device), labels.to(device)
            # print(f"Batch: {batch_idx}")
            labels, low_res, high_res = labels.to(device), low_res.to(device), high_res.to(device)

            optimizer.zero_grad()

            if low_res_input:
                inputs = low_res
            else:
                inputs = high_res

            outputs = classifier(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * low_res.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        epoch_f1 = f1_score(labels.cpu(), predicted.cpu(), average='macro')
        # Save checkpoint after each epoch
        training_loop.save_classifier_checkpoint(epoch, classifier, optimizer,
                                                 running_loss, checkpoint_path,
                                                 model_name)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f} | F1 Score: {epoch_f1:.4f}")






def train_classifier_w_generator(classifier, 
                                 generator, 
                                 optimizer, 
                                 criterion, 
                                 data_loader, 
                                 num_epochs, 
                                 device, 
                                 checkpoint_path, 
                                 model_name):
    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (labels, low_res, high_res) in enumerate(data_loader):
            #images, labels = images.to(device), labels.to(device)
            # print(f"Batch: {batch_idx}")
            labels, low_res, high_res = labels.to(device), low_res.to(device), high_res.to(device)

            optimizer.zero_grad()

            # WE WANT GENERATOR OUTPUT TO BE USED AS THE INPUT
            # outputs = classifier(GENERATOR_OUTPUT)
            # FORWARD PASS W GENERATOR = LOW RES -> FAKE HIGH RES
            with torch.no_grad(): # DONT TRAIN THE GENERATOR
              fake_images = generator(low_res)
            outputs = classifier(fake_images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * outputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        epoch_f1 = f1_score(labels.cpu(), predicted.cpu(), average='macro')
        # Save checkpoint after each epoch
        training_loop.save_classifier_checkpoint(epoch, classifier, optimizer,
                                                 running_loss, checkpoint_path,
                                                 model_name)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f} | F1 Score: {epoch_f1:.4f}")


def generate_f1_report(classifier, data_loader, device, low_res_input=True):
    classifier.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for labels, low_res, high_res in data_loader:
            labels, low_res, high_res = labels.to(device), low_res.to(device), high_res.to(device)
            if low_res_input:
                outputs = classifier(low_res)
            else:
                outputs = classifier(high_res)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    report = classification_report(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    print(f"Overall F1 Score: {f1:.4f}")


def generate_f1_report_w_generator(classifier, generator, data_loader, device):
    classifier.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for labels, low_res, high_res in data_loader:
            labels, low_res, high_res = labels.to(device), low_res.to(device), high_res.to(device)
            # FORWARD PASS W GENERATOR = LOW RES -> FAKE HIGH RES
            fake_images = generator(low_res)
            outputs = classifier(fake_images)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    report = classification_report(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    print(f"Overall F1 Score: {f1:.4f}")