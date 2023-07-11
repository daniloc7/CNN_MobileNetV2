import os
import shutil
import random

class PatchSeparator:
    def __init__(self, diretorio_treinamento, diretorio_validacao):
        self.diretorio_treinamento = diretorio_treinamento
        self.diretorio_validacao = diretorio_validacao

    def separar_arquivos_validacao(self, diretorio_origem, num_arquivos):
        arquivos = os.listdir(diretorio_origem)
        random.shuffle(arquivos)
        arquivos_validacao = arquivos[:num_arquivos]

        for arquivo in os.listdir(self.diretorio_validacao):
            os.remove(os.path.join(self.diretorio_validacao, arquivo))

        for arquivo in arquivos_validacao:
            origem = os.path.join(diretorio_origem, arquivo)
            destino = os.path.join(self.diretorio_validacao, arquivo)
            shutil.move(origem, destino)

# import os
# import random
# import shutil
# import config as cfg


# class PatchSeparator:
#     def __init__(self, diretorio_treinamento, diretorio_validacao):
#         self.diretorio_treinamento = diretorio_treinamento
#         self.diretorio_validacao = diretorio_validacao

#     def separar_arquivos_validacao(self, diretorio_origem, num_arquivos):
#         arquivos = os.listdir(diretorio_origem)
#         random.shuffle(arquivos)
#         arquivos_validacao = arquivos[:num_arquivos]

#         for arquivo in arquivos_validacao:
#             origem = os.path.join(diretorio_origem, arquivo)
#             destino = os.path.join(self.diretorio_validacao, arquivo)
#             shutil.move(origem, destino)
