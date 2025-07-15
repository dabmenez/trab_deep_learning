import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_training_curves(train_losses, val_losses, val_accuracies, save_path='training_curves.png'):
    """Plota curvas de treinamento para apresentação"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Curvas de Treinamento - ModelNet10 com GCN', fontsize=16, fontweight='bold')
    
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
    axes[1, 0].plot(train_losses, label='Train Loss', color='blue', alpha=0.7, linewidth=2)
    axes[1, 0].plot(val_losses, label='Val Loss', color='red', alpha=0.7, linewidth=2)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Loss (log scale)')
    axes[1, 0].set_title('Loss em Escala Logarítmica')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Resumo das métricas finais
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
    
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    axes[1, 1].set_title('Resumo do Treinamento')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plota a matriz de confusão para apresentação"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão - ModelNet10', fontsize=16, fontweight='bold')
    plt.xlabel('Predição', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_class_performance(y_true, y_pred, class_names, save_path='class_performance.png'):
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
