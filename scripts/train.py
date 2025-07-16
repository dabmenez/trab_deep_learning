import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import FaceToEdge, RandomRotate
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import time
from tqdm import tqdm
from model.gcn import GCN3DClassifier, GCN
from utils.normalizer import normalize
from utils.plots import plot_training_curves, plot_class_performance, plot_confusion_matrix
from utils.docs import save_training_metrics, generate_final_report
import os
import numpy as np

data_path = "./"  # Caminho relativo para salvar arquivos

def setup_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_modelnet_dataset():
    """Carrega o dataset ModelNet10 com data augmentation"""
    # Caminho correto para o root do dataset (diretamente na pasta ModelNet10)
    dataset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ModelNet10'))

    # Data augmentation para treino
    train_transform = FaceToEdge(remove_faces=False)
    val_transform = FaceToEdge(remove_faces=False)

    train_dataset = ModelNet(root=dataset_root, name='10', train=True, transform=train_transform)
    val_dataset = ModelNet(root=dataset_root, name='10', train=False, transform=val_transform)

    return normalize(train_dataset), normalize(val_dataset)


def validate(model, val_loader, criterion, device, epoch):
    """Valida o modelo"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f"Epoch: {epoch} Validation"):
            data = data.to(device)
            out = model(data)
                
            loss = criterion(out, data.y)
            total_loss += loss.item()
            num_batches += 1
            
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_labels.append(data.y.cpu())           
        
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = accuracy_score(labels, preds)
    # CORREÇÃO: dividir pelo número de batches, não pelo tamanho do dataset
    return total_loss / num_batches, acc, preds, labels


def calculate_class_weights(dataset):
    """Calcula pesos para classes desbalanceadas"""
    class_counts = {}
    for data in dataset:
        label = data.y.item()
        class_counts[label] = class_counts.get(label, 0) + 1
    
    total_samples = len(dataset)
    class_weights = torch.zeros(10)  # 10 classes no ModelNet10
    
    for class_idx in range(10):
        if class_idx in class_counts:
            class_weights[class_idx] = total_samples / (len(class_counts) * class_counts[class_idx])
        else:
            class_weights[class_idx] = 1.0
    
    return class_weights


def main():
    # Inicializando variaveis - AJUSTADAS PARA GPU PEQUENA
    batch_size = 8   # 4x maior que o original
    epochs = 10   # Mantido
    lr = 5e-4        # Mantido
    patience = 20    # Mantido
    best_val_acc = patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    device = setup_device()
    print(f"Usando dispositivo: {device}")
    
    train_dataset, val_dataset = load_modelnet_dataset()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Usar modelo ajustado para GPU pequena
    model = GCN3DClassifier(
        input_dim=3, 
        hidden_dim=64,  # Reduzido de 128 para 64 (ainda melhor que o original)
        num_classes=10, 
        dropout=0.3
    ).to(device)

    print(f"Parâmetros do modelo: {sum(p.numel() for p in model.parameters()):,}")
    
    # Calcular class weights para lidar com desbalanceamento
    class_weights = calculate_class_weights(train_dataset).to(device)
    print(f"Class weights: {class_weights}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Treinamento
    start_time = time.time()
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_batches = 0

        for data in tqdm(train_loader, desc=f"Epoch: {epoch} Training"):
            data = data.to(device)
            output = model(data)

            loss = criterion(output, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        train_loss = total_loss / num_batches
        
        # Validação
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device, epoch)
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        scheduler.step()
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