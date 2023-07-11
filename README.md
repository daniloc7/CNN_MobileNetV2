# CNN_MobileNetV2
CNN identifica se é uma fita preta ou branca com uma determinada acurácia.
Código criado para o relatório técnico da matéria Fundamentos de Inteligência Artificial do Mestrado em Computação Aplicada UEPG

1 - run.py: é responsável por criar a rede neural MobileNetV2 e extrair as caracteristicas do dataset, porém, ele não as classifica,
a ultima camada(softmax) é removida. Para fazer o nosso classificador, foi selecionado a camada Flatten do MobileNetV2, acrescentado mais 2 camadas densas, alternando camadas de dropout de 20% entre elas. Neste mesmo código é realizado a poda e também apresenta um gráfico 
acurácia x épocas.

2 - patch.generator.py: realiza o recorte das imagens originais para patches retangulares de dimensões 250x200.

3 - patch.separator.py: após o recorte das imagens originais, o patch.separator determina cerca de 20%(3 patches) para irem para a pasta de
validação.

4 - test.py: é selecionado uma imagem aleatória da pasta valid para visualizar como está a precisão da rede neural treinado. Exemplo de
output: [[0.65795904 0.342041  ]], significa que a rede neural entrega com o nivel de confiança de 65,79% que a imagem fornecida pertence a classe 1(fitabranca).

5 - config.py: classe criada para manter as configurações estáticas. 
