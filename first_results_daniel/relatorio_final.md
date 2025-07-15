
# RELATÓRIO FINAL - CLASSIFICAÇÃO 3D COM GCN

## 🎯 Resultados Principais
- **Acurácia de Teste**: 0.7633 (76.33%)
- **F1-Score**: 0.7473
- **Melhor Acurácia de Validação**: 0.7843 (78.43%)
- **Épocas Treinadas**: 200
- **Tempo de Treinamento**: 39.2 minutos
- **Device Utilizado**: GPU

## 📈 Análise de Convergência
- **Loss Final (Treinamento)**: 0.7244
- **Loss Final (Validação)**: 0.6829
- **Época de Melhor Performance**: 199

## 🧠 Arquitetura do Modelo
- **Tipo**: Graph Convolutional Network (GCN)
- **Camadas Convolucionais**: 3 camadas GCN
- **Pooling**: Global Mean Pooling
- **Camadas Lineares**: 2 camadas fully connected
- **Dropout**: 0.3 para regularização
- **Dataset**: ModelNet10 (3991 amostras, 10 classes)

## 📊 Performance
- **Overfitting**: Não
- **Convergência**: Instável
- **Eficiência**: Regular

## 📁 Arquivos Gerados
- `best_model.pth` - Melhor modelo treinado
- `training_curves.png` - Curvas de treinamento
- `confusion_matrix.png` - Matriz de confusão
- `class_performance.png` - Performance por classe
- `training_metrics.json` - Métricas detalhadas

## 🎓 Conclusões
O modelo GCN demonstrou boa capacidade de classificação de objetos 3D, 
alcançando uma acurácia de 76.3% no conjunto de teste. 
A arquitetura com múltiplas camadas convolucionais permitiu capturar 
características hierárquicas dos grafos 3D de forma eficiente.

---
*Gerado em: 14/07/2025 19:38:53*
