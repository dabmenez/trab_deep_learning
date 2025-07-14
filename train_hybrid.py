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
import time
import json
from datetime import datetime

# Configuração para reprodutibilidade
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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

def setup_device_with_fallback():
    """Configura device com fallback automático GPU -> CPU"""
    if torch.cuda.is_available():
        try:
            # Testar se CUDA funciona
            test_tensor = torch.randn(10, 10).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            
            # Verificar memória disponível
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
            print(f"   Memória GPU: {gpu_memory:.1f} GB")
            
            if gpu_memory < 4.0:  # Se menos de 4GB, usar configurações otimizadas
                print("⚠️  GPU com memória limitada - usando configurações otimizadas")
                return torch.device('cuda'), True  # True = GPU otimizada
            else:
                return torch.device('cuda'), False  # False = GPU normal
                
        except Exception as e:
            print(f"⚠️  Erro na GPU: {e}")
            print("🔄 Fallback para CPU")
            return torch.device('cpu'), False
    else:
        print("🔄 Usando CPU")
        return torch.device('cpu'), False

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
    print("Normalizando dados...")
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

def create_dataloaders(train_dataset, val_dataset, test_dataset, device_type, optimized=False):
    """Cria DataLoaders com configurações otimizadas para o device"""
    if device_type == 'cuda' and optimized:
        # Configurações otimizadas para GPU com memória limitada
        batch_size = 8
        hidden_dim = 32
        print(f"📊 Configurações otimizadas: batch_size={batch_size}, hidden_dim={hidden_dim}")
    elif device_type == 'cuda':
        # Configurações normais para GPU
        batch_size = 16
        hidden_dim = 64
        print(f"📊 Configurações normais: batch_size={batch_size}, hidden_dim={hidden_dim}")
    else:
        # Configurações para CPU
        batch_size = 32
        hidden_dim = 64
        print(f"📊 Configurações CPU: batch_size={batch_size}, hidden_dim={hidden_dim}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, hidden_dim

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Treina o modelo por uma época"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.pos, data.edge_index, data.batch)
        loss = criterion(out, data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        
        # Limpar cache GPU se necessário
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return total_loss / num_batches

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

def train_model(device, train_loader, val_loader, test_loader, hidden_dim, num_classes, device_name):
    """Treina o modelo e retorna métricas"""
    print(f"\n🚀 Iniciando treinamento com {device_name}...")
    
    # Modelo
    model = GCN3DClassifier(
        input_dim=3, 
        hidden_dim=hidden_dim, 
        num_classes=num_classes, 
        dropout=0.3
    ).to(device)
    
    print(f"Modelo criado: {num_classes} classes, {hidden_dim} hidden dim")
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
            model_filename = f'best_model_{device_name.lower()}.pth'
            torch.save(model.state_dict(), model_filename)
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
    model_filename = f'best_model_{device_name.lower()}.pth'
    if os.path.exists(model_filename):
        model.load_state_dict(torch.load(model_filename))
        print("✅ Melhor modelo carregado")
    
    # Teste final
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    
    # Métricas detalhadas
    report = classification_report(test_labels, test_preds, output_dict=True)
    f1_score = report['weighted avg']['f1-score']
    
    print(f'\n📊 RESULTADOS FINAIS ({device_name})')
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1-Score: {f1_score:.4f}')
    print(f'Best Val Accuracy: {best_val_acc:.4f}')
    
    return {
        'device_name': device_name,
        'training_time': training_time,
        'test_accuracy': test_acc,
        'test_f1_score': f1_score,
        'best_val_accuracy': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'test_preds': test_preds,
        'test_labels': test_labels,
        'model': model,
        'hidden_dim': hidden_dim
    }

def plot_comparison_results(gpu_results, cpu_results):
    """Plota comparação entre GPU e CPU"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparação GPU vs CPU - ModelNet10 com GCN', fontsize=16, fontweight='bold')
    
    # 1. Loss comparison
    axes[0, 0].plot(gpu_results['train_losses'], label=f'GPU Train', color='blue', alpha=0.7)
    axes[0, 0].plot(gpu_results['val_losses'], label=f'GPU Val', color='red', alpha=0.7)
    axes[0, 0].plot(cpu_results['train_losses'], label=f'CPU Train', color='green', alpha=0.7)
    axes[0, 0].plot(cpu_results['val_losses'], label=f'CPU Val', color='orange', alpha=0.7)
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Comparação de Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy comparison
    axes[0, 1].plot(gpu_results['val_accuracies'], label=f'GPU ({gpu_results["test_accuracy"]:.3f})', color='blue', linewidth=2)
    axes[0, 1].plot(cpu_results['val_accuracies'], label=f'CPU ({cpu_results["test_accuracy"]:.3f})', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Acurácia')
    axes[0, 1].set_title('Comparação de Acurácia')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Training time comparison
    devices = ['GPU', 'CPU']
    times = [gpu_results['training_time']/60, cpu_results['training_time']/60]
    colors = ['blue', 'green']
    bars = axes[0, 2].bar(devices, times, color=colors, alpha=0.7)
    axes[0, 2].set_ylabel('Tempo (minutos)')
    axes[0, 2].set_title('Tempo de Treinamento')
    for bar, time_val in zip(bars, times):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       f'{time_val:.1f}min', ha='center', va='bottom')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Final metrics comparison
    metrics = ['Test Acc', 'F1-Score', 'Best Val Acc']
    gpu_metrics = [gpu_results['test_accuracy'], gpu_results['test_f1_score'], gpu_results['best_val_accuracy']]
    cpu_metrics = [cpu_results['test_accuracy'], cpu_results['test_f1_score'], cpu_results['best_val_accuracy']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, gpu_metrics, width, label='GPU', color='blue', alpha=0.7)
    axes[1, 0].bar(x + width/2, cpu_metrics, width, label='CPU', color='green', alpha=0.7)
    axes[1, 0].set_xlabel('Métricas')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Comparação de Métricas Finais')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Speedup calculation
    speedup = cpu_results['training_time'] / gpu_results['training_time']
    axes[1, 1].bar(['Speedup'], [speedup], color='purple', alpha=0.7)
    axes[1, 1].set_ylabel('Speedup (CPU/GPU)')
    axes[1, 1].set_title(f'GPU é {speedup:.1f}x mais rápida')
    axes[1, 1].text(0, speedup + 0.1, f'{speedup:.1f}x', ha='center', va='bottom', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary text
    summary_text = f"""
    RESUMO DA COMPARAÇÃO:
    
    GPU ({gpu_results['device_name']}):
    • Tempo: {gpu_results['training_time']/60:.1f} min
    • Acurácia: {gpu_results['test_accuracy']:.3f}
    • F1-Score: {gpu_results['test_f1_score']:.3f}
    • Hidden Dim: {gpu_results['hidden_dim']}
    
    CPU:
    • Tempo: {cpu_results['training_time']/60:.1f} min
    • Acurácia: {cpu_results['test_accuracy']:.3f}
    • F1-Score: {cpu_results['test_f1_score']:.3f}
    • Hidden Dim: {cpu_results['hidden_dim']}
    
    Speedup: {speedup:.1f}x
    """
    
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    axes[1, 2].set_title('Resumo da Comparação')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_gpu_cpu.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_comparison_metrics(gpu_results, cpu_results):
    """Salva métricas de comparação"""
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "gpu_results": {
            "device_name": gpu_results['device_name'],
            "training_time": gpu_results['training_time'],
            "test_accuracy": float(gpu_results['test_accuracy']),
            "test_f1_score": float(gpu_results['test_f1_score']),
            "best_val_accuracy": float(gpu_results['best_val_accuracy']),
            "hidden_dim": gpu_results['hidden_dim']
        },
        "cpu_results": {
            "device_name": "CPU",
            "training_time": cpu_results['training_time'],
            "test_accuracy": float(cpu_results['test_accuracy']),
            "test_f1_score": float(cpu_results['test_f1_score']),
            "best_val_accuracy": float(cpu_results['best_val_accuracy']),
            "hidden_dim": cpu_results['hidden_dim']
        },
        "comparison": {
            "speedup": float(cpu_results['training_time'] / gpu_results['training_time']),
            "accuracy_difference": float(gpu_results['test_accuracy'] - cpu_results['test_accuracy']),
            "f1_difference": float(gpu_results['test_f1_score'] - cpu_results['test_f1_score'])
        }
    }
    
    with open('comparison_metrics.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("✅ Métricas de comparação salvas em: comparison_metrics.json")

def main():
    """Função principal - Treina com GPU e CPU"""
    print("=== CLASSIFICAÇÃO 3-D COM GCN - MODELNET10 (GPU + CPU) ===")
    
    # Configurar seed
    set_seed(42)
    
    # Configurar device com fallback
    device, optimized = setup_device_with_fallback()
    device_name = "GPU" if device.type == 'cuda' else "CPU"
    
    # Carregar dataset
    data_list, num_classes = load_modelnet_dataset()
    
    # Dividir dataset
    train_dataset, val_dataset, test_dataset = split_dataset(data_list, seed=42)
    
    # Criar DataLoaders
    train_loader, val_loader, test_loader, hidden_dim = create_dataloaders(
        train_dataset, val_dataset, test_dataset, device.type, optimized
    )
    
    # Treinar com device atual (GPU ou CPU)
    current_results = train_model(device, train_loader, val_loader, test_loader, 
                                hidden_dim, num_classes, device_name)
    
    # Se treinou com GPU, treinar também com CPU para comparação
    if device.type == 'cuda':
        print("\n" + "="*50)
        print("🔄 Treinando também com CPU para comparação...")
        print("="*50)
        
        # Configurar CPU
        cpu_device = torch.device('cpu')
        
        # Criar DataLoaders para CPU (configurações normais)
        cpu_train_loader, cpu_val_loader, cpu_test_loader, cpu_hidden_dim = create_dataloaders(
            train_dataset, val_dataset, test_dataset, 'cpu', False
        )
        
        # Treinar com CPU
        cpu_results = train_model(cpu_device, cpu_train_loader, cpu_val_loader, cpu_test_loader,
                                cpu_hidden_dim, num_classes, "CPU")
        
        # Comparar resultados
        print("\n" + "="*50)
        print("📊 COMPARAÇÃO GPU vs CPU")
        print("="*50)
        
        speedup = cpu_results['training_time'] / current_results['training_time']
        acc_diff = current_results['test_accuracy'] - cpu_results['test_accuracy']
        
        print(f"⏱️  Speedup GPU: {speedup:.1f}x mais rápida")
        print(f"📈 Diferença de acurácia: {acc_diff:.4f} ({'GPU melhor' if acc_diff > 0 else 'CPU melhor'})")
        print(f"🎯 GPU: {current_results['test_accuracy']:.4f} | CPU: {cpu_results['test_accuracy']:.4f}")
        
        # Gerar visualizações de comparação
        print("\n📊 Gerando gráficos de comparação...")
        plot_comparison_results(current_results, cpu_results)
        
        # Salvar métricas de comparação
        save_comparison_metrics(current_results, cpu_results)
        
        print("\n🎉 COMPARAÇÃO CONCLUÍDA!")
        print("📁 Arquivos gerados:")
        print("   - comparison_gpu_cpu.png")
        print("   - comparison_metrics.json")
        print("   - best_model_gpu.pth")
        print("   - best_model_cpu.pth")
    
    else:
        print("\n🎉 Treinamento apenas com CPU concluído!")
        print("📁 Arquivo gerado: best_model_cpu.pth")

if __name__ == "__main__":
    main() 