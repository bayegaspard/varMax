import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
)
from sklearn.metrics import ConfusionMatrixDisplay

class Evaluator:
    def __init__(self, model, test_loader, label_encoder, unknown_label, device="cpu", results_dir="/home/offsec/llm/USMA/adv_rob/defence/results"):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.label_encoder = label_encoder
        self.unknown_label = int(unknown_label)  # Ensure unknown_label is an integer
        self.device = device
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def evaluate_and_plot_metrics(self, description="Test_Data"):
        all_labels = []
        all_predictions = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                predictions = logits.argmax(dim=1).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())

        metrics = {
            'Accuracy': accuracy_score(all_labels, all_predictions),
            'F1 Score': f1_score(all_labels, all_predictions, average='weighted', zero_division=1),
            'Precision': precision_score(all_labels, all_predictions, average='weighted', zero_division=1),
            'Recall': recall_score(all_labels, all_predictions, average='weighted', zero_division=1)
        }

        # Plot metrics as a bar chart
        plt.figure(figsize=(8, 5))
        plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'salmon', 'limegreen', 'orange'])
        plt.title(f'{description} - Metrics')
        plt.ylabel('Score')
        plt.tight_layout()
        metrics_path = os.path.join(self.results_dir, f'{description}_metrics.png')
        plt.savefig(metrics_path)
        plt.close()
        print(f"Metrics visualization saved as '{metrics_path}'")

        # Classification report
        report = classification_report(
            all_labels, all_predictions, labels=list(range(self.unknown_label + 1)),
            target_names=list(self.label_encoder.classes_) + ["Unknown"], zero_division=1
        )
        report_path = os.path.join(self.results_dir, f"{description}_classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Classification report saved as '{report_path}'")

        # Confusion matrix
        class_names = list(self.label_encoder.classes_) + ["Unknown"]
        cm = confusion_matrix(all_labels, all_predictions, labels=range(self.unknown_label + 1))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix ({description})', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=45, va='center', fontsize=10)
        plt.tight_layout()
        cm_path = os.path.join(self.results_dir, f'confusion_matrix_{description}.png')
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Confusion matrix saved as '{cm_path}'")
