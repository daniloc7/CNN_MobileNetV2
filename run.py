import matplotlib.pyplot as plt
import tempfile
import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_model_optimization as tfmot
from keras.optimizers import Adam
from keras.models import Model
from keras.applications import MobileNet
from keras.layers import Flatten, Dense, Dropout
from patch_generator import PatchGenerator
from patch_separator import PatchSeparator
import config as cfg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# recortar patches
PatchGenerator
generator = PatchGenerator(cfg.diretorio_fita_branca,
                           cfg.diretorio_treinamento + r'/fitabranca', cfg.largura_patch, cfg.altura_patch)
generator.criar_patches()
generator = PatchGenerator(
    cfg.diretorio_fita_preta, cfg.diretorio_treinamento + r'/fitapreta', cfg.largura_patch, cfg.altura_patch)
generator.criar_patches()

# separar patches para pasta valid
patch_separator = PatchSeparator(
    cfg.diretorio_treinamento, cfg.diretorio_validacao + r'/fitabranca')
patch_separator.separar_arquivos_validacao(
    os.path.join(cfg.diretorio_treinamento, 'fitabranca'), 3)

patch_separator = PatchSeparator(
    cfg.diretorio_treinamento, cfg.diretorio_validacao + r'/fitapreta')
patch_separator.separar_arquivos_validacao(
    os.path.join(cfg.diretorio_treinamento, 'fitapreta'), 3)

# augmentation com mais mudanças
datagen = ImageDataGenerator(rescale=1/255.0,
                             rotation_range=30,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest'
                             )

# # augmentation menor
# datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True
#                              )

classes = ['fitabranca', 'fitapreta']
target_size = (cfg.largura_patch, cfg.altura_patch)
class_mode = 'categorical'
num_classes = len(classes)

# train_datagen vai receber as imagens alteradas pelo augmentation
train_datagen = datagen.flow_from_directory(
    cfg.diretorio_treinamento,
    target_size=target_size,
    batch_size=cfg.batch_size,
    class_mode=class_mode,
    classes=classes
)

validation_dataset = datagen.flow_from_directory(batch_size=cfg.batch_size,
                                                 directory=cfg.diretorio_validacao,
                                                 shuffle=True,
                                                 target_size=target_size,
                                                 #  subset="validation",
                                                 class_mode=class_mode,
                                                 classes=classes)

print("Nomes das classes:", train_datagen.class_indices)

# (include_top) camada de classificacao não sera incluida, só vamos extrair as caracteristicas para alimentar outra camada de rede
base_model = MobileNet(
    include_top=False, weights='imagenet', input_shape=(250, 200, 3))
print(base_model.output_shape)
flatten_output = base_model.output

x = Flatten()(flatten_output)
x = Dropout(0.2)(x)  # camada dropout, taxa de 20%. (reduz o overfitting)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)  # camada dropout, taxa de 20%.
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(
    x)  # camada softmax é o classificador,

modelo = Model(inputs=base_model.input, outputs=output)

# visualizar as camadas do modelo
# for i, layer in enumerate(modelo.layers):
#     # 86 é a flatten(caracteristicas), após tem mais 2 camadas densas(relu), e depois a camada densa de saida(softmax)
#     print(i, layer.name)

# for layer in modelo.layers:
#     layer.trainable = False

# realizar poda
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                             final_sparsity=0.8,
                                                             begin_step=0,
                                                             end_step=cfg.epochs * len(train_datagen))
}

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

model_for_pruning = prune_low_magnitude(modelo, **pruning_params)

model_for_pruning.compile(optimizer=Adam(learning_rate=cfg.learning_rate),
                          #   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

model_for_pruning.summary()


logdir = tempfile.mkdtemp()

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

history = model_for_pruning.fit(train_datagen,
                                batch_size=cfg.batch_size, steps_per_epoch=len(train_datagen), epochs=cfg.epochs, validation_data=validation_dataset,
                                validation_steps=len(validation_dataset),
                                callbacks=callbacks)


val_loss, val_accuracy = model_for_pruning.evaluate(validation_dataset)
print('Acurácia no conjunto de validação:', val_accuracy)

plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
