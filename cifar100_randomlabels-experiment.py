import deel.lip as lip
import deel.lip.layers as lip_lay
import deel.lip.losses as lip_losses
import dlt.data.loader as loader
import dlt.data.pipeline as pipeline
import dlt.infrastructure.distributed_training as distributed
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
from deel.lip.compute_layer_sv import compute_model_upper_lip
from dlt.model_factory import *
from wandb.keras import WandbCallback
import wandb

wandb.init(
    project="understand_your_loss",
    entity="ananymized",
    name="cifar100_randomlabels",
    save_code=True,
)
config = wandb.config
###########################################################################
# load data and randomize labels
###########################################################################
ds_train, ds_test, metadata = loader.get_cifar100()
ds_train = ds_train.map(
    lambda x, y: (
        x,
        tf.random.uniform([], maxval=metadata["nb_classes"], seed=612, dtype=tf.int64),
    )
).cache()
ds_test = ds_test.map(
    lambda x, y: (
        x,
        tf.random.uniform([], maxval=metadata["nb_classes"], seed=612, dtype=tf.int64),
    )
).cache()
###########################################################################
# logged hyperparameters
###########################################################################
config.bs = 1000
config.epochs = 500
config.lr = 1e-3
config.loss = "HKR"  # must be in ["crossentropy", "cosine_similarity", "HKR"]
###########################################################################
# data preparation and augmentation
###########################################################################
ds_train, ds_test = pipeline.prepare_data(
    ds_train,
    ds_test,
    preparation_x=[lambda x: tf.cast(x, tf.float32) / 255],
    preparation_y=[lambda y: tf.one_hot(y, metadata["nb_classes"])],
    augmentation_x=[],
    batch_size=config.bs,
)
###########################################################################
# build model
###########################################################################
# return the correct strategy for CPU/GPU/multi GPU/TPU depending on available hardware
strategy = distributed.get_distribution_strategy()
with strategy.scope():
    model = tf.keras.models.Sequential(
        [
            layers.Input(metadata["input_shape"]),
            layers.Flatten(),
            lip_lay.SpectralDense(
                1024, use_bias=True, activation="deel-lip>GroupSort2"
            ),
            lip_lay.SpectralDense(
                1024, use_bias=True, activation="deel-lip>GroupSort2"
            ),
            lip_lay.SpectralDense(
                metadata["nb_classes"],
                use_bias=True,
                activation=None,
                # disjoint_neurons=True,
            ),
        ]
    )
    if config.loss == "HKR":
        config.alpha = 256
        config.min_margin = 36 / 255
        loss = lip_losses.MulticlassHKR(
            alpha=config.alpha, min_margin=config.min_margin
        )
    elif config.loss == "crossentropy":
        config.tau = 256.0
        loss = (
            lambda yt, yp: losses.categorical_crossentropy(
                yt, config.tau * yp, from_logits=True
            )
            / config.tau
        )
    elif config.loss == "cosine_similarity":
        loss = losses.CosineSimilarity()

    model.compile(
        loss=loss,
        metrics=[
            "accuracy",
            lip.metrics.CategoricalProvableRobustAccuracy(
                epsilon=36 / 255, disjoint_neurons=False, name="rob_acc_36"
            ),
            lip.metrics.CategoricalProvableRobustAccuracy(
                epsilon=72 / 255, disjoint_neurons=False, name="rob_acc_72"
            ),
            lip.metrics.CategoricalProvableAvgRobustness(
                disjoint_neurons=False, name="avg_rob"
            ),
        ],
        optimizer=optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecay(
                config.lr,
                config.epochs * metadata["nb_samples_train"] // config.bs,
                1e-2,
            )
        ),
    )
tf.keras.utils.plot_model(model, show_shapes=True)
model.summary()
###########################################################################
# fit model
###########################################################################
model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=config.epochs,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint("cifar100_randomlabels.h5"),
        WandbCallback(),
    ],
)
layers_sv, model_sv = compute_model_upper_lip(model)
wandb.log({"lipschitz_upper_cste": np.prod([high for low, high in model_sv.values()])})
