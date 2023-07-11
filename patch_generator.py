import os
from PIL import Image

class PatchGenerator:
    def __init__(self, diretorio, diretorio_patches, largura_patch, altura_patch):
        self.diretorio = diretorio
        self.diretorio_patches = diretorio_patches
        self.largura_patch = largura_patch
        self.altura_patch = altura_patch

    def criar_patches(self):
        if not os.path.exists(self.diretorio_patches):
            os.makedirs(self.diretorio_patches)

        for arquivo in os.listdir(self.diretorio):
            if arquivo.endswith('.tif') or arquivo.endswith('.png'):
                caminho_imagem = os.path.join(self.diretorio, arquivo)
                imagem = Image.open(caminho_imagem)

                largura_imagem, altura_imagem = imagem.size

                num_patches_largura = largura_imagem // self.largura_patch
                num_patches_altura = altura_imagem // self.altura_patch

                for i in range(num_patches_largura):
                    for j in range(num_patches_altura):
                        left = i * self.largura_patch
                        upper = j * self.altura_patch
                        right = (i + 1) * self.largura_patch
                        lower = (j + 1) * self.altura_patch

                        patch = imagem.crop((left, upper, right, lower))

                        nome_patch = f'{arquivo}_patch_{i}_{j}.tif'
                        caminho_patch = os.path.join(
                            self.diretorio_patches, nome_patch)
                        patch.save(caminho_patch)
