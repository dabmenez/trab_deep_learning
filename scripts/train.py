import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import FaceToEdge
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import time
from model.gcn import GCN3DClassifier
from utils.normalizer import normalize
from utils.plots import plot_training_curves, plot_class_performance, plot_confusion_matrix
from utils.docs import save_training_metrics, generate_final_report

data_path = ""

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
    print(f"Dataset de treinamento: {len(train_dataset)} amostras, {train_dataset.num_classes} classes")

    val_dataset = ModelNet(root='data/ModelNet10', name='10', train=False, transform=FaceToEdge(remove_faces=False))
    print(f"Dataset de validação: {len(val_dataset)} amostras, {val_dataset.num_classes} classes")

    return normalize(train_dataset), normalize(val_dataset)


def validate(model, loader, criterion, device):
    """Valida o modelo"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    with torch.no_grad():
        for data in loader:
                
            data = data.to(device)
            
            try:
                out = model(data.pos, data.edge_index, data.batch)
                
                # Verificar se a saída e target têm dimensões compatíveis
                if out.size(0) != data.y.size(0):
                    continue
                    
                # Garantir que y seja 1D
                y_target = data.y.squeeze()
                if y_target.dim() == 0:
                    y_target = y_target.unsqueeze(0)
                    
                loss = criterion(out, y_target)
                total_loss += loss.item()
                
                pred = out.argmax(dim=1)
                all_preds.append(pred.cpu())
                all_labels.append(y_target.cpu())
                num_batches += 1
                
            except Exception as e:
                continue
    
    if len(all_preds) == 0:
        return 0.0, 0.0, torch.tensor([]), torch.tensor([])
        
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = accuracy_score(labels, preds)
    return total_loss / max(num_batches, 1), acc, preds, labels


def main():
    batch_size = 16
    epochs = 10
    device = setup_device()
    
    train_dataset, val_dataset = load_modelnet_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GCN3DClassifier(
        input_dim=3, 
        hidden_dim=64, 
        num_classes=10, 
        dropout=0.3
    ).to(device)

    print(f"Parâmetros do modelo: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # try without weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    criterion = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.MSELoss()
    
    # Listas para métricas
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Early stopping
    best_val_acc = 0
    patience = 30
    patience_counter = 0
    
    # Treinamento
    start_time = time.time()
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for data in train_loader:
            data = data.to(device)
            output = model(data.pos, data.edge_index, data.batch)
                
            # Garantir que y seja 1D ?
            y_target = data.y.squeeze()
            if y_target.dim() == 0:
                y_target = y_target.unsqueeze(0)

            loss = loss_fn(output, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() # *batch_size?

        train_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:>2}/{epochs} — Train MSE Loss: {train_loss:.6f}")
        
        # Validação
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, loss_fn, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Armazenar métricas
        train_losses.append()
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
        
        if epoch % 10 == 0:
            print(f'Época {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    training_time = time.time() - start_time
    print(f"⏱️  Tempo total de treinamento: {training_time/60:.1f} minutos")
    
    # Carregar melhor modelo
    model.load_state_dict(torch.load(data_path+'best_model.pth'))


    # Métricas detalhadas
    report = classification_report(val_labels, val_preds, output_dict=True)
    f1_score = report['weighted avg']['f1-score']
    
    print('\n📊 RESULTADOS FINAIS')
    print(f'Test Accuracy: {val_acc:.4f}')
    print(f'Test F1-Score: {f1_score:.4f}')
    print(f'Best Val Accuracy: {best_val_acc:.4f}')
    
    # Relatório detalhado
    class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    print("\nRelatório de Classificação:")
    print(classification_report(val_labels, val_preds, target_names=class_names))
    
    # Gerar visualizações para apresentação
    print("\n📊 Gerando visualizações para apresentação...")
    
    # Documentar 
    plot_training_curves(train_losses, val_losses, val_accuracies, data_path+'training_curves.png')
    plot_confusion_matrix(val_labels, val_preds, class_names, data_path+'confusion_matrix.png')
    plot_class_performance(val_labels, val_preds, class_names, data_path+'class_performance.png')
    save_training_metrics(train_losses, val_losses, val_accuracies, val_acc, f1_score, 
                          best_val_acc, len(train_losses), device, hidden_dim=64)
    generate_final_report(val_acc, f1_score, best_val_acc, len(train_losses),
                         train_losses, val_losses, val_accuracies, device, training_time)
    
    # Análise final
    print("\n🎯 ANÁLISE FINAL")
    print(f"Melhor acurácia de validação: {best_val_acc:.4f}")
    print(f"Época de melhor performance: {np.argmax(val_accuracies) + 1}")
    print(f"Loss final de treinamento: {train_losses[-1]:.4f}")
    print(f"Loss final de validação: {val_losses[-1]:.4f}")
    print(f"Tempo total: {training_time/60:.1f} minutos")
 

if __name__ == "__main__":
    main() 