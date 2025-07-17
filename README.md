# Classificação 3D com Graph Convolutional Networks

Este projeto implementa classificação de objetos 3D usando Graph Convolutional Networks (GCN) com PyG.\
**Autores:** Daniel Menezes & Gabriela Casini

## Dataset

- **ModelNet10**: ~5000 amostras, 10 classes
- **Classes**: bathtub, bed, chair, desk, dresser, monitor, night_stand, sofa, table, toilet
- **Formato**: Malhas 3D convertidas para grafos


## Como fazer seus próprios experimentos

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Ajustar o caminho para o projeto

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/trab_deep_learning" 
```

### 3. Executar treinamento

```bash
python scripts/train.py
```

**Arquivos gerados:**
- `best_model.pth` - Melhor modelo
- `training_curves.png` - Curvas de treinamento
- `confusion_matrix.png` - Matriz de confusão
- `class_performance.png` - Performance por classe
- `training_metrics.json` - Métricas detalhadas
- `relatorio_final.md` - Relatório Final


## Principais Referências

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [ModelNet Dataset](https://modelnet.cs.princeton.edu/)
- [Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
