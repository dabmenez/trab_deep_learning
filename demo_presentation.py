#!/usr/bin/env python3
"""
Script de demonstração rápida para testar funcionalidades de apresentação
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

def create_demo_metrics():
    """Cria métricas de demonstração para testar os gráficos"""
    print("🎭 Criando métricas de demonstração...")
    
    # Simular dados de treinamento
    epochs = 50
    np.random.seed(42)
    
    # CPU - convergência mais lenta
    cpu_train_losses = [2.5 * np.exp(-epoch/30) + 0.1 * np.random.randn() for epoch in range(epochs)]
    cpu_val_losses = [2.2 * np.exp(-epoch/25) + 0.15 * np.random.randn() for epoch in range(epochs)]
    cpu_val_accuracies = [0.3 + 0.4 * (1 - np.exp(-epoch/20)) + 0.02 * np.random.randn() for epoch in range(epochs)]
    
    # GPU - convergência mais rápida
    gpu_train_losses = [2.5 * np.exp(-epoch/20) + 0.08 * np.random.randn() for epoch in range(epochs)]
    gpu_val_losses = [2.2 * np.exp(-epoch/18) + 0.12 * np.random.randn() for epoch in range(epochs)]
    gpu_val_accuracies = [0.3 + 0.5 * (1 - np.exp(-epoch/15)) + 0.015 * np.random.randn() for epoch in range(epochs)]
    
    # Garantir que acurácias estejam entre 0 e 1
    cpu_val_accuracies = np.clip(cpu_val_accuracies, 0, 1)
    gpu_val_accuracies = np.clip(gpu_val_accuracies, 0, 1)
    
    # Métricas finais
    cpu_test_acc = 0.67
    gpu_test_acc = 0.78
    cpu_f1 = 0.65
    gpu_f1 = 0.76
    
    # Salvar métricas CPU
    cpu_metrics = {
        "device_type": "CPU",
        "timestamp": datetime.now().isoformat(),
        "training_metrics": {
            "train_losses": cpu_train_losses,
            "val_losses": cpu_val_losses,
            "val_accuracies": cpu_val_accuracies.tolist()
        },
        "final_results": {
            "test_accuracy": cpu_test_acc,
            "test_f1_score": cpu_f1,
            "best_val_accuracy": max(cpu_val_accuracies),
            "epochs_trained": epochs,
            "final_train_loss": cpu_train_losses[-1],
            "final_val_loss": cpu_val_losses[-1]
        },
        "model_info": {
            "architecture": "GCN3DClassifier",
            "input_dim": 3,
            "hidden_dim": 64,
            "num_classes": 10,
            "dropout": 0.3
        }
    }
    
    # Salvar métricas GPU
    gpu_metrics = {
        "device_type": "GPU",
        "timestamp": datetime.now().isoformat(),
        "training_metrics": {
            "train_losses": gpu_train_losses,
            "val_losses": gpu_val_losses,
            "val_accuracies": gpu_val_accuracies.tolist()
        },
        "final_results": {
            "test_accuracy": gpu_test_acc,
            "test_f1_score": gpu_f1,
            "best_val_accuracy": max(gpu_val_accuracies),
            "epochs_trained": epochs,
            "final_train_loss": gpu_train_losses[-1],
            "final_val_loss": gpu_val_losses[-1]
        },
        "model_info": {
            "architecture": "GCN3DClassifierGPU",
            "input_dim": 3,
            "hidden_dim": 128,
            "num_classes": 10,
            "dropout": 0.3,
            "num_layers": 3
        }
    }
    
    # Salvar arquivos
    with open('training_metrics_cpu.json', 'w') as f:
        json.dump(cpu_metrics, f, indent=2)
    
    with open('training_metrics_gpu.json', 'w') as f:
        json.dump(gpu_metrics, f, indent=2)
    
    print("✅ Métricas de demonstração criadas!")
    return cpu_metrics, gpu_metrics

def create_demo_confusion_matrices():
    """Cria matrizes de confusão de demonstração"""
    print("📊 Criando matrizes de confusão de demonstração...")
    
    class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    
    # Simular matriz de confusão CPU (menos precisa)
    np.random.seed(42)
    cpu_cm = np.random.randint(20, 80, size=(10, 10))
    np.fill_diagonal(cpu_cm, np.random.randint(60, 90, size=10))
    
    # Simular matriz de confusão GPU (mais precisa)
    gpu_cm = np.random.randint(15, 60, size=(10, 10))
    np.fill_diagonal(gpu_cm, np.random.randint(70, 95, size=10))
    
    # Normalizar
    cpu_cm = cpu_cm / cpu_cm.sum(axis=1, keepdims=True) * 100
    gpu_cm = gpu_cm / gpu_cm.sum(axis=1, keepdims=True) * 100
    
    # Plotar CPU
    plt.figure(figsize=(12, 10))
    sns.heatmap(cpu_cm, annot=True, fmt='.0f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão - ModelNet10 (CPU) - DEMO', fontsize=16, fontweight='bold')
    plt.xlabel('Predição', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_cpu.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plotar GPU
    plt.figure(figsize=(12, 10))
    sns.heatmap(gpu_cm, annot=True, fmt='.0f', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão - ModelNet10 (GPU) - DEMO', fontsize=16, fontweight='bold')
    plt.xlabel('Predição', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_gpu.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Matrizes de confusão criadas!")

def create_demo_training_curves():
    """Cria curvas de treinamento de demonstração"""
    print("📈 Criando curvas de treinamento de demonstração...")
    
    # Carregar métricas
    with open('training_metrics_cpu.json', 'r') as f:
        cpu_metrics = json.load(f)
    with open('training_metrics_gpu.json', 'r') as f:
        gpu_metrics = json.load(f)
    
    cpu_train_losses = cpu_metrics['training_metrics']['train_losses']
    cpu_val_losses = cpu_metrics['training_metrics']['val_losses']
    cpu_val_accuracies = cpu_metrics['training_metrics']['val_accuracies']
    
    gpu_train_losses = gpu_metrics['training_metrics']['train_losses']
    gpu_val_losses = gpu_metrics['training_metrics']['val_losses']
    gpu_val_accuracies = gpu_metrics['training_metrics']['val_accuracies']
    
    # CPU curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Curvas de Treinamento - ModelNet10 com GCN (CPU) - DEMO', fontsize=16, fontweight='bold')
    
    # 1. Loss vs Época
    axes[0, 0].plot(cpu_train_losses, label='Treinamento', color='blue', linewidth=2)
    axes[0, 0].plot(cpu_val_losses, label='Validação', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Evolução da Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Acurácia vs Época
    axes[0, 1].plot(cpu_val_accuracies, label='Validação', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Acurácia')
    axes[0, 1].set_title('Evolução da Acurácia')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Loss em escala logarítmica
    axes[0, 2].plot(cpu_train_losses, label='Train Loss', color='blue', alpha=0.7, linewidth=2)
    axes[0, 2].plot(cpu_val_losses, label='Val Loss', color='red', alpha=0.7, linewidth=2)
    axes[0, 2].set_yscale('log')
    axes[0, 2].set_xlabel('Época')
    axes[0, 2].set_ylabel('Loss (log scale)')
    axes[0, 2].set_title('Loss em Escala Logarítmica')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Indicador de overfitting
    loss_diff = [abs(train - val) for train, val in zip(cpu_train_losses, cpu_val_losses)]
    axes[1, 0].plot(loss_diff, color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('|Train Loss - Val Loss|')
    axes[1, 0].set_title('Indicador de Overfitting')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Convergência
    axes[1, 1].plot(cpu_val_accuracies, color='green', linewidth=2)
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Acurácia')
    axes[1, 1].set_title('Convergência da Acurácia')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Resumo
    summary_text = f"""
    Métricas Finais (DEMO):
    
    Loss Final (Train): {cpu_train_losses[-1]:.4f}
    Loss Final (Val): {cpu_val_losses[-1]:.4f}
    Acurácia Final (Val): {cpu_val_accuracies[-1]:.4f}
    Melhor Acurácia (Val): {max(cpu_val_accuracies):.4f}
    Épocas Treinadas: {len(cpu_train_losses)}
    """
    
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=12, verticalalignment='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    axes[1, 2].set_title('Resumo do Treinamento')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_curves_cpu.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # GPU curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Curvas de Treinamento - ModelNet10 com GCN (GPU) - DEMO', fontsize=16, fontweight='bold')
    
    # 1. Loss vs Época
    axes[0, 0].plot(gpu_train_losses, label='Treinamento', color='blue', linewidth=2)
    axes[0, 0].plot(gpu_val_losses, label='Validação', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Evolução da Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Acurácia vs Época
    axes[0, 1].plot(gpu_val_accuracies, label='Validação', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Acurácia')
    axes[0, 1].set_title('Evolução da Acurácia')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Loss em escala logarítmica
    axes[0, 2].plot(gpu_train_losses, label='Train Loss', color='blue', alpha=0.7, linewidth=2)
    axes[0, 2].plot(gpu_val_losses, label='Val Loss', color='red', alpha=0.7, linewidth=2)
    axes[0, 2].set_yscale('log')
    axes[0, 2].set_xlabel('Época')
    axes[0, 2].set_ylabel('Loss (log scale)')
    axes[0, 2].set_title('Loss em Escala Logarítmica')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Indicador de overfitting
    loss_diff = [abs(train - val) for train, val in zip(gpu_train_losses, gpu_val_losses)]
    axes[1, 0].plot(loss_diff, color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('|Train Loss - Val Loss|')
    axes[1, 0].set_title('Indicador de Overfitting')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Convergência
    axes[1, 1].plot(gpu_val_accuracies, color='green', linewidth=2)
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Acurácia')
    axes[1, 1].set_title('Convergência da Acurácia')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Resumo
    summary_text = f"""
    Métricas Finais (DEMO):
    
    Loss Final (Train): {gpu_train_losses[-1]:.4f}
    Loss Final (Val): {gpu_val_losses[-1]:.4f}
    Acurácia Final (Val): {gpu_val_accuracies[-1]:.4f}
    Melhor Acurácia (Val): {max(gpu_val_accuracies):.4f}
    Épocas Treinadas: {len(gpu_train_losses)}
    """
    
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=12, verticalalignment='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    axes[1, 2].set_title('Resumo do Treinamento')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_curves_gpu.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Curvas de treinamento criadas!")

def create_demo_reports():
    """Cria relatórios de demonstração"""
    print("📝 Criando relatórios de demonstração...")
    
    # Carregar métricas
    with open('training_metrics_cpu.json', 'r') as f:
        cpu_metrics = json.load(f)
    with open('training_metrics_gpu.json', 'r') as f:
        gpu_metrics = json.load(f)
    
    # Relatório CPU
    cpu_report = f"""
# RELATÓRIO DE TREINAMENTO - MODELNET10 COM GCN (CPU) - DEMO

## 📊 Resultados Finais
- **Acurácia de Teste**: {cpu_metrics['final_results']['test_accuracy']:.4f} ({cpu_metrics['final_results']['test_accuracy']*100:.2f}%)
- **F1-Score**: {cpu_metrics['final_results']['test_f1_score']:.4f}
- **Melhor Acurácia de Validação**: {cpu_metrics['final_results']['best_val_accuracy']:.4f} ({cpu_metrics['final_results']['best_val_accuracy']*100:.2f}%)
- **Épocas Treinadas**: {cpu_metrics['final_results']['epochs_trained']}

## 📈 Análise de Convergência
- **Loss Final (Treinamento)**: {cpu_metrics['final_results']['final_train_loss']:.4f}
- **Loss Final (Validação)**: {cpu_metrics['final_results']['final_val_loss']:.4f}

## 🎯 Performance
- **Overfitting**: Não
- **Convergência**: Estável

## 📁 Arquivos Gerados
- `best_model_cpu.pth` - Melhor modelo
- `training_curves_cpu.png` - Curvas de treinamento
- `confusion_matrix_cpu.png` - Matriz de confusão
- `training_metrics_cpu.json` - Métricas detalhadas

---
*Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - DEMO*
"""
    
    # Relatório GPU
    gpu_report = f"""
# RELATÓRIO DE TREINAMENTO - MODELNET10 COM GCN (GPU) - DEMO

## 📊 Resultados Finais
- **Acurácia de Teste**: {gpu_metrics['final_results']['test_accuracy']:.4f} ({gpu_metrics['final_results']['test_accuracy']*100:.2f}%)
- **F1-Score**: {gpu_metrics['final_results']['test_f1_score']:.4f}
- **Melhor Acurácia de Validação**: {gpu_metrics['final_results']['best_val_accuracy']:.4f} ({gpu_metrics['final_results']['best_val_accuracy']*100:.2f}%)
- **Épocas Treinadas**: {gpu_metrics['final_results']['epochs_trained']}

## 📈 Análise de Convergência
- **Loss Final (Treinamento)**: {gpu_metrics['final_results']['final_train_loss']:.4f}
- **Loss Final (Validação)**: {gpu_metrics['final_results']['final_val_loss']:.4f}

## 🚀 Otimizações GPU
- **Batch Size**: 64 (vs 32 no CPU)
- **Hidden Dim**: 128 (vs 64 no CPU)
- **Camadas GCN**: 3 (vs 2 no CPU)
- **Otimizador**: AdamW (vs Adam no CPU)

## 📁 Arquivos Gerados
- `best_model_gpu.pth` - Melhor modelo
- `training_curves_gpu.png` - Curvas de treinamento
- `confusion_matrix_gpu.png` - Matriz de confusão
- `training_metrics_gpu.json` - Métricas detalhadas

---
*Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - DEMO*
"""
    
    # Salvar relatórios
    with open('relatorio_cpu.md', 'w', encoding='utf-8') as f:
        f.write(cpu_report)
    
    with open('relatorio_gpu.md', 'w', encoding='utf-8') as f:
        f.write(gpu_report)
    
    print("✅ Relatórios criados!")

def main():
    """Função principal da demonstração"""
    print("🎭 === DEMONSTRAÇÃO DAS FUNCIONALIDADES DE APRESENTAÇÃO ===")
    print("Este script cria dados simulados para demonstrar os gráficos e relatórios.")
    print("Execute os treinamentos reais para obter resultados verdadeiros.\n")
    
    # Criar métricas de demonstração
    create_demo_metrics()
    
    # Criar visualizações
    create_demo_confusion_matrices()
    create_demo_training_curves()
    create_demo_reports()
    
    # Executar relatório final
    print("\n🔄 Executando relatório final...")
    try:
        from generate_final_report import create_comprehensive_comparison
        create_comprehensive_comparison()
    except ImportError:
        print("⚠️  Script generate_final_report.py não encontrado")
    
    print("\n🎉 DEMONSTRAÇÃO CONCLUÍDA!")
    print("\n📁 Arquivos criados:")
    print("  - training_metrics_cpu.json")
    print("  - training_metrics_gpu.json")
    print("  - training_curves_cpu.png")
    print("  - training_curves_gpu.png")
    print("  - confusion_matrix_cpu.png")
    print("  - confusion_matrix_gpu.png")
    print("  - relatorio_cpu.md")
    print("  - relatorio_gpu.md")
    print("  - comprehensive_comparison.png (se disponível)")
    print("  - performance_analysis.png (se disponível)")
    print("  - executive_summary.png (se disponível)")
    print("  - relatorio_final.md (se disponível)")
    
    print("\n💡 Para resultados reais, execute:")
    print("  python train_with_modelnet.py")
    print("  python train_gpu_optimized.py")
    print("  python generate_final_report.py")

if __name__ == "__main__":
    main() 