import torch
from lib.data_loader import DataLoaderModule
from lib.model import VarMaxClassifier
from lib.trainer import Trainer
from lib.evaluator import Evaluator
from lib.varmax import VarMaxScorer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader_module = DataLoaderModule()
    train_loader, test_loader, label_encoder, num_classes, unknown_label, scaler = DataLoaderModule().load_data()

    #train_loader, test_loader, label_encoder, num_classes, _ = data_loader_module.load_data()
    
    input_size = next(iter(train_loader))[0].shape[1]
    model = VarMaxClassifier(input_size=input_size, hidden_size=64, num_classes=num_classes).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, train_loader, criterion, optimizer, device)
    trainer.train(num_epochs=10)

    # evaluator = Evaluator(model, test_loader, label_encoder, device)
    evaluator = Evaluator(model, test_loader, label_encoder, unknown_label, device)

    evaluator.evaluate_and_plot_metrics(description="Test_Data")

    varmax_scorer = VarMaxScorer(model, device=device)
    varmax_scores, ground_truth = varmax_scorer.compute_scores(test_loader)
    # print("VarMax Scores:", varmax_scores)
    # print("Ground Truth Labels:", ground_truth)

if __name__ == "__main__":
    main()
