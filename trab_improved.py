import torch
import torch.nn.functional as F
from torch_geometric.datasets import GeometricShapes
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Configuração para reprodutibilidade
torch.manual_seed(42)
np.random.seed(42)

# Carregamento e preparação dos dados
dataset = GeometricShapes(root='data/GeometricShapes')
dataset = dataset.shuffle()

# Divisão em treinamento, validação e teste
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:train_size + val_size]
test_dataset = dataset[train_size + val_size:]

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class ImprovedGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
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
        x = self.dropout(x)
        
        # Terceira camada convolucional
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Pooling global
        x = global_mean_pool(x, batch)
        
        # Camadas lineares com dropout
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x

# Inicialização do modelo
model = ImprovedGCN(input_dim=2, hidden_dim=64, num_classes=3, dropout=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
criterion = torch.nn.CrossEntropyLoss()

# Listas para armazenar métricas
train_losses = []
val_losses = []
val_accuracies = []

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            
            pred = out.argmax(dim=1)
            all_preds.append(pred)
            all_labels.append(data.y)
    
    acc = accuracy_score(torch.cat(all_labels), torch.cat(all_preds))
    return total_loss / len(loader), acc

def test(loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            all_preds.append(pred)
            all_labels.append(data.y)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    
    return acc, all_preds, all_labels

# Treinamento com early stopping
best_val_acc = 0
patience = 20
patience_counter = 0

print("Iniciando treinamento...")
for epoch in range(1, 201):
    # Treinamento
    train_loss = train()
    
    # Validação
    val_loss, val_acc = validate(val_loader)
    
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
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping na época {epoch}")
        break
    
    if epoch % 10 == 0:
        print(f'Época {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Carregar melhor modelo
model.load_state_dict(torch.load('best_model.pth'))

# Teste final
test_acc, test_preds, test_labels = test(test_loader)
print(f'\nResultado Final - Acurácia no Teste: {test_acc:.4f}')

# Relatório de classificação detalhado
print("\nRelatório de Classificação:")
print(classification_report(test_labels, test_preds))

# Plotagem das métricas
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Treinamento', color='blue')
plt.plot(val_losses, label='Validação', color='red')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.title('Evolução da Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validação', color='green')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.title('Evolução da Acurácia')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# Análise de convergência
print(f"\nAnálise de Convergência:")
print(f"Melhor acurácia de validação: {best_val_acc:.4f}")
print(f"Época de melhor performance: {np.argmax(val_accuracies) + 1}")
print(f"Loss final de treinamento: {train_losses[-1]:.4f}")
print(f"Loss final de validação: {val_losses[-1]:.4f}")

# Verificação de overfitting
if train_losses[-1] < val_losses[-1] * 0.5:
    print("⚠️  Possível overfitting detectado!")
else:
    print("✅ Modelo parece bem generalizado") 