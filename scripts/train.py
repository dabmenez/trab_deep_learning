import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import FaceToEdge
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import time
from tqdm import tqdm
from model.gcn import GCN3DClassifier, GCN
from utils.normalizer import normalize
from utils.plots import plot_training_curves, plot_class_performance, plot_confusion_matrix
from utils.docs import save_training_metrics, generate_final_report

data_path = "/Users/gccasini/Downloads/"

def setup_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_modelnet_dataset():
    """Carrega o dataset ModelNet10"""

    train_dataset = ModelNet(root='data/ModelNet10', name='10', train=True, transform=FaceToEdge(remove_faces=False))
    val_dataset = ModelNet(root='data/ModelNet10', name='10', train=False, transform=FaceToEdge(remove_faces=False))

    return normalize(train_dataset), normalize(val_dataset)


def validate(model, val_loader, criterion, device, epoch):
    """Valida o modelo"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f"Epoch: {epoch} Validation"):

            data = data.to(device)
            out = model(data)
                
            loss = criterion(out, data.y)
            total_loss += loss.item()
            
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_labels.append(data.y.cpu())           
        
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = accuracy_score(labels, preds)
    return total_loss / len(val_loader.dataset), acc, preds, labels


def main():
    # Inicializando variaveis
    batch_size = 2
    epochs = 10
    lr = 1e-3
    patience = 30
    best_val_acc = patience_counter = 0
    train_losses = val_losses = val_accuracies = []
    device = setup_device()
    
    train_dataset, val_dataset = load_modelnet_dataset()

    # Sample pra testar no meu pc
    train_dataset = train_dataset[0:100]
    val_dataset = val_dataset[0:50]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # model = GCN3DClassifier(
    #     input_dim=3, 
    #     hidden_dim=64, 
    #     num_classes=10, 
    #     dropout=0.3
    # ).to(device)

    model = GCN(input_ch=3, num_classes=10).to(device)

    print(f"Parâmetros do modelo: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # try without weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    criterion = torch.nn.CrossEntropyLoss()

    # Treinamento
    start_time = time.time()
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for data in tqdm(train_loader, desc=f"Epoch: {epoch} Training"):
            data = data.to(device)
            output = model(data)

            loss = criterion(output, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        
        # Validação
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device, epoch)
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Salvar melhor modelo
            torch.save(model.state_dict(), data_path+'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping na época {epoch}")
            break

    
    training_time = time.time() - start_time
    
    # Carregar melhor modelo
    model.load_state_dict(torch.load(data_path+'best_model.pth'))

    # Métricas detalhadas
    report = classification_report(val_labels, val_preds, output_dict=True)
    f1_score = report['weighted avg']['f1-score']
    
    # Relatório detalhado
    class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    print("\nRelatório de Classificação:")
    print(classification_report(val_labels, val_preds, target_names=class_names))
    
    # Documentar 
    plot_training_curves(train_losses, val_losses, val_accuracies, data_path+'training_curves.png')
    plot_confusion_matrix(val_labels, val_preds, class_names, data_path+'confusion_matrix.png')
    plot_class_performance(val_labels, val_preds, class_names, data_path+'class_performance.png')
    save_training_metrics(train_losses, val_losses, val_accuracies, val_acc, f1_score, 
                          best_val_acc, len(train_losses), device.type, hidden_dim=64)
    generate_final_report(val_acc, f1_score, best_val_acc, len(train_losses),
                         train_losses, val_losses, val_accuracies, device.type, training_time)

if __name__ == "__main__":
    main() 