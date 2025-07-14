import torch
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import FaceToEdge, NormalizeScale
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from torch.utils.data import random_split
import json
import time
from datetime import datetime

# Configuração para reprodutibilidade
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class GCN3DClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = torch.nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x, edge_index, batch):
        # Primeira camada convolucional
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Segunda camada convolucional
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Pooling global
        x = global_mean_pool(x, batch)
        
        # Camadas lineares
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x

def load_modelnet_dataset():
    """Carrega o dataset ModelNet10"""
    print("Carregando ModelNet10 dataset...")
    
    # Transformações
    transform = FaceToEdge(remove_faces=False)
    normalize_transform = NormalizeScale()
    
    # Carregar dataset
    dataset = ModelNet(root='data/ModelNet10', name='10', train=True, transform=transform)
    
    print(f"Dataset carregado: {len(dataset)} amostras, {dataset.num_classes} classes")
    
    # Normalizar coordenadas (criar nova lista)
    normalized_data_list = [normalize_transform(data) for data in dataset]
    
    # Verificar distribuição de classes
    all_labels = [data.y.item() for data in normalized_data_list]
    label_counts = {}
    for label in all_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"Distribuição de classes: {label_counts}")
    
    return normalized_data_list, dataset.num_classes

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Divide o dataset usando random_split"""
    assert train_ratio + val_ratio + test_ratio == 1.0
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Split sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Treina o modelo por uma época"""
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.pos, data.edge_index, data.batch)
        loss = criterion(out, data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, loader, criterion, device):
    """Valida o modelo"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.pos, data.edge_index, data.batch)
            loss = criterion(out, data.y.squeeze())
            total_loss += loss.item()
            
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_labels.append(data.y.squeeze().cpu())
    
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = accuracy_score(labels, preds)
    return total_loss / len(loader), acc, preds, labels

def plot_comprehensive_training_curves(train_losses, val_losses, val_accuracies, save_path='training_curves_cpu.png'):
    """Plota curvas de treinamento completas para apresentação"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Curvas de Treinamento - ModelNet10 com GCN (CPU)', fontsize=16, fontweight='bold')
    
    # 1. Loss vs Época
    axes[0, 0].plot(train_losses, label='Treinamento', color='blue', linewidth=2)
    axes[0, 0].plot(val_losses, label='Validação', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Evolução da Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Acurácia vs Época
    axes[0, 1].plot(val_accuracies, label='Validação', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Acurácia')
    axes[0, 1].set_title('Evolução da Acurácia')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Loss em escala logarítmica
    axes[0, 2].plot(train_losses, label='Train Loss', color='blue', alpha=0.7, linewidth=2)
    axes[0, 2].plot(val_losses, label='Val Loss', color='red', alpha=0.7, linewidth=2)
    axes[0, 2].set_yscale('log')
    axes[0, 2].set_xlabel('Época')
    axes[0, 2].set_ylabel('Loss (log scale)')
    axes[0, 2].set_title('Loss em Escala Logarítmica')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Diferença entre train e val loss (overfitting)
    loss_diff = [abs(train - val) for train, val in zip(train_losses, val_losses)]
    axes[1, 0].plot(loss_diff, color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('|Train Loss - Val Loss|')
    axes[1, 0].set_title('Indicador de Overfitting')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Taxa de aprendizado (se disponível)
    axes[1, 1].plot(val_accuracies, color='green', linewidth=2)
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Acurácia')
    axes[1, 1].set_title('Convergência da Acurácia')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Resumo das métricas finais
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    final_val_acc = val_accuracies[-1]
    best_val_acc = max(val_accuracies)
    
    summary_text = f"""
    Métricas Finais:
    
    Loss Final (Train): {final_train_loss:.4f}
    Loss Final (Val): {final_val_loss:.4f}
    Acurácia Final (Val): {final_val_acc:.4f}
    Melhor Acurácia (Val): {best_val_acc:.4f}
    Épocas Treinadas: {len(train_losses)}
    """
    
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=12, verticalalignment='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    axes[1, 2].set_title('Resumo do Treinamento')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix_cpu.png'):
    """Plota a matriz de confusão para apresentação"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão - ModelNet10 (CPU)', fontsize=16, fontweight='bold')
    plt.xlabel('Predição', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_performance(y_true, y_pred, class_names, save_path='class_performance_cpu.png'):
    """Plota performance por classe"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico de barras para precision, recall, f1
    x = np.arange(len(class_names))
    width = 0.25
    
    ax1.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax1.bar(x, recall, width, label='Recall', alpha=0.8)
    ax1.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance por Classe')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de suporte (número de amostras)
    ax2.bar(class_names, support, alpha=0.7, color='orange')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Número de Amostras')
    ax2.set_title('Distribuição de Amostras por Classe')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_training_metrics(train_losses, val_losses, val_accuracies, test_acc, f1_score, 
                         best_val_acc, epochs_trained, device_type="CPU"):
    """Salva métricas de treinamento em JSON"""
    metrics = {
        "device_type": device_type,
        "timestamp": datetime.now().isoformat(),
        "training_metrics": {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        },
        "final_results": {
            "test_accuracy": float(test_acc),
            "test_f1_score": float(f1_score),
            "best_val_accuracy": float(best_val_acc),
            "epochs_trained": epochs_trained,
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(val_losses[-1])
        },
        "model_info": {
            "architecture": "GCN3DClassifier",
            "input_dim": 3,
            "hidden_dim": 64,
            "num_classes": 10,
            "dropout": 0.3
        }
    }
    
    filename = f"training_metrics_{device_type.lower()}.json"
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Métricas salvas em: {filename}")

def generate_presentation_report(test_acc, f1_score, best_val_acc, epochs_trained, 
                               train_losses, val_losses, device_type="CPU"):
    """Gera relatório para apresentação"""
    report = f"""
# RELATÓRIO DE TREINAMENTO - MODELNET10 COM GCN ({device_type})

## 📊 Resultados Finais
- **Acurácia de Teste**: {test_acc:.4f} ({test_acc*100:.2f}%)
- **F1-Score**: {f1_score:.4f}
- **Melhor Acurácia de Validação**: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)
- **Épocas Treinadas**: {epochs_trained}

## 📈 Análise de Convergência
- **Loss Final (Treinamento)**: {train_losses[-1]:.4f}
- **Loss Final (Validação)**: {val_losses[-1]:.4f}
- **Época de Melhor Performance**: {np.argmax(val_accuracies) + 1}

## 🎯 Performance
- **Overfitting**: {'Sim' if train_losses[-1] < val_losses[-1] * 0.5 else 'Não'}
- **Convergência**: {'Estável' if val_losses[-1] < 0.5 else 'Instável'}

## 📁 Arquivos Gerados
- `best_model_{device_type.lower()}.pth` - Melhor modelo
- `training_curves_{device_type.lower()}.png` - Curvas de treinamento
- `confusion_matrix_{device_type.lower()}.png` - Matriz de confusão
- `class_performance_{device_type.lower()}.png` - Performance por classe
- `training_metrics_{device_type.lower()}.json` - Métricas detalhadas

---
*Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*
"""
    
    filename = f"relatorio_{device_type.lower()}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Relatório salvo em: {filename}")

def main():
    """Função principal"""
    print("=== CLASSIFICAÇÃO 3-D COM GCN - MODELNET10 (CPU) ===")
    
    # Configurar seed
    set_seed(42)
    
    # Configurar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    # Carregar dataset
    data_list, num_classes = load_modelnet_dataset()
    
    # Dividir dataset
    train_dataset, val_dataset, test_dataset = split_dataset(data_list, seed=42)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Modelo
    model = GCN3DClassifier(
        input_dim=3, 
        hidden_dim=64, 
        num_classes=num_classes, 
        dropout=0.3
    ).to(device)
    
    print(f"Modelo criado: {num_classes} classes")
    print(f"Parâmetros do modelo: {sum(p.numel() for p in model.parameters()):,}")
    
    # Otimizador e critério
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Listas para métricas
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Early stopping
    best_val_acc = 0
    patience = 30
    patience_counter = 0
    
    # Treinamento
    print("\nIniciando treinamento...")
    start_time = time.time()
    
    for epoch in range(1, 201):
        # Treinamento
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validação
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Armazenar métricas
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Salvar melhor modelo
            torch.save(model.state_dict(), 'best_model_cpu.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping na época {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f'Época {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    training_time = time.time() - start_time
    print(f"\nTempo total de treinamento: {training_time/60:.1f} minutos")
    
    # Carregar melhor modelo
    if os.path.exists('best_model_cpu.pth'):
        model.load_state_dict(torch.load('best_model_cpu.pth'))
        print("Melhor modelo carregado")
    
    # Teste final
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    
    # Métricas detalhadas
    report = classification_report(test_labels, test_preds, output_dict=True)
    f1_score = report['weighted avg']['f1-score']
    
    print(f'\n=== RESULTADOS FINAIS (CPU) ===')
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1-Score: {f1_score:.4f}')
    print(f'Best Val Accuracy: {best_val_acc:.4f}')
    
    # Relatório detalhado
    class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    print(f"\nRelatório de Classificação:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Gerar visualizações para apresentação
    print("\n=== GERANDO VISUALIZAÇÕES PARA APRESENTAÇÃO ===")
    
    # Plotar curvas completas
    plot_comprehensive_training_curves(train_losses, val_losses, val_accuracies, 'training_curves_cpu.png')
    
    # Plotar matriz de confusão
    plot_confusion_matrix(test_labels, test_preds, class_names, 'confusion_matrix_cpu.png')
    
    # Plotar performance por classe
    plot_class_performance(test_labels, test_preds, class_names, 'class_performance_cpu.png')
    
    # Salvar métricas
    save_training_metrics(train_losses, val_losses, val_accuracies, test_acc, f1_score, 
                         best_val_acc, len(train_losses), "CPU")
    
    # Gerar relatório
    generate_presentation_report(test_acc, f1_score, best_val_acc, len(train_losses),
                               train_losses, val_losses, "CPU")
    
    # Análise de convergência
    print(f"\n=== ANÁLISE DE CONVERGÊNCIA (CPU) ===")
    print(f"Melhor acurácia de validação: {best_val_acc:.4f}")
    print(f"Época de melhor performance: {np.argmax(val_accuracies) + 1}")
    print(f"Loss final de treinamento: {train_losses[-1]:.4f}")
    print(f"Loss final de validação: {val_losses[-1]:.4f}")
    print(f"Tempo total: {training_time/60:.1f} minutos")
    
    # Verificação de overfitting
    if train_losses[-1] < val_losses[-1] * 0.5:
        print("⚠️  Possível overfitting detectado!")
    else:
        print("✅ Modelo parece bem generalizado")
    
    print(f"\n🎉 TREINAMENTO CONCLUÍDO! Arquivos gerados para apresentação.")

if __name__ == "__main__":
    main() 