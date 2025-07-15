import numpy as np
import json
from datetime import datetime

def save_training_metrics(train_losses, val_losses, val_accuracies, test_acc, f1_score, 
                         best_val_acc, epochs_trained, device_name, hidden_dim):
    """Salva métricas de treinamento em JSON"""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "device_used": device_name,
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
            "hidden_dim": hidden_dim,
            "num_classes": 10,
            "dropout": 0.3,
            "num_conv_layers": 3
        }
    }
    
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("✅ Métricas salvas em: training_metrics.json")


def generate_final_report(test_acc, f1_score, best_val_acc, epochs_trained, 
                         train_losses, val_losses, val_accuracies, device_name, training_time):
    """Gera relatório final para apresentação"""
    report = f"""
# RELATÓRIO FINAL - CLASSIFICAÇÃO 3D COM GCN

## 🎯 Resultados Principais
- **Acurácia de Teste**: {test_acc:.4f} ({test_acc*100:.2f}%)
- **F1-Score**: {f1_score:.4f}
- **Melhor Acurácia de Validação**: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)
- **Épocas Treinadas**: {epochs_trained}
- **Tempo de Treinamento**: {training_time/60:.1f} minutos
- **Device Utilizado**: {device_name}

## 📈 Análise de Convergência
- **Loss Final (Treinamento)**: {train_losses[-1]:.4f}
- **Loss Final (Validação)**: {val_losses[-1]:.4f}
- **Época de Melhor Performance**: {np.argmax(val_accuracies) + 1}

## 📊 Performance
- **Overfitting**: {'Sim' if train_losses[-1] < val_losses[-1] * 0.5 else 'Não'}
- **Convergência**: {'Estável' if val_losses[-1] < 0.5 else 'Instável'}
- **Eficiência**: {'Boa' if test_acc > 0.8 else 'Regular' if test_acc > 0.7 else 'Baixa'}

"""
    
    with open('relatorio_final.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ Relatório salvo em: relatorio_final.md")
