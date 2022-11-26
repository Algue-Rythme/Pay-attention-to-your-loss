import argparse
import deel.lip as lip
import dlt.data.loader as loader
import dlt.data.pipeline as pipeline
import dlt.infrastructure.distributed_training as distributed
import foolbox as fb
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
from dlt.model_factory import *
from wandb.keras import WandbCallback
import wandb

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tau",
    dest="tau",
    type=float,
    help="crossentropy_temp",
    default=1.0,
)
parser.add_argument(
    "--perc",
    dest="perc",
    type=float,
    help="percentage of dataset",
    default=1.0,
)
args = parser.parse_args()

###########################################################################
# declaration of wand hparams
###########################################################################

wandb.init(
    project="understand_your_loss-consistency",
    entity="ananymized",
    name=f"CCE{args.tau}_{args.perc}ds",
    save_code=True,
)

config = wandb.config
config.bs = 1000
config.epochs = 300
config.lr = 1e-5
config.loss = "crossentropy"
config.perc_dataset = args.perc

###########################################################################
# load data and randomize labels
###########################################################################
ds_train, ds_test, metadata = loader.get_cifar10()
repeats = int(1.0 / config.perc_dataset)
# repeats = 5
ds_train = ds_train.take(int(metadata["nb_samples_train"] * config.perc_dataset))

nb_step_per_epoch = int(metadata["nb_samples_train"] * config.perc_dataset) // config.bs
###########################################################################
# data preparation and augmentation
###########################################################################
ds_train, ds_test = pipeline.prepare_data(
    ds_train,
    ds_test,
    preparation_x=[lambda x: tf.cast(x, tf.float32) / 255.0],
    preparation_y=[lambda y: tf.one_hot(y, metadata["nb_classes"])],
    augmentation_x=[],
    batch_size=config.bs,
)
ds_train = ds_train.repeat(repeats)
###########################################################################
# build model
###########################################################################
# return the correct strategy for CPU/GPU/multi GPU/TPU depending on available hardware
strategy = distributed.get_distribution_strategy()
with strategy.scope():
    kwargs = dict(
        conv_sizes=(
            (32, 32),
            (64, 64),
            (128, 128),
        ),
        dense_sizes=(128,),
    )
    ds_neurons = False
    layer_params = dict(
        # conv=utils.ClassParam(
        #     dlt_lay.soc_conv.SOCConv2D,
        #     kernel_size=(3, 3),
        #     use_bias=True,
        # ),
        conv=utils.ClassParam(
            lip.layers.OrthoConv2D,
            kernel_size=(3, 3),
            eps_spectral=1e-7,
            use_bias=True,
            regul_lorth=0.0,
        ),
        # conv=utils.ClassParam(lip.layers.SpectralConv2D, kernel_size=(3, 3), eps_bjorck=None, beta_bjorck=None, use_bias=False,),
        dense=utils.ClassParam(lip.layers.SpectralDense, use_bias=True),
        # last_dense=utils.ClassParam(
        #     lip.layers.FrobeniusDense,
        #     disjoint_neurons=True,
        #     activation=None,
        #     use_bias=True,
        # ),
        last_dense=utils.ClassParam(
            lip.layers.SpectralDense,
            eps_bjorck=None,
            beta_bjorck=None,
            activation=None,
            use_bias=True,
        ),
        # pooling=None,
        pooling=utils.ClassParam(
            lip.layers.InvertibleDownSampling, pool_size=(2, 2)
        ),  # lip.layers.ScaledL2NormPooling2D,
        global_pooling=utils.ClassParam(
            layers.Flatten,
            # lip.layers.ScaledGlobalL2NormPooling2D
        ),
        # activation=utils.ClassParam(dlt_lay.soc_conv.HouseHolder),
        activation=utils.ClassParam(lip.activations.GroupSort2),
    )
    kwargs.update(layer_params)
    # call the factory and build the model
    model = vgg.VGG(metadata["input_shape"], metadata["nb_classes"], **kwargs)

    if config.loss == "crossentropy":
        config.tau = args.tau
        loss = lip.losses.TauCategoricalCrossentropy(config.tau)

    model.compile(
        loss=loss,
        metrics=[
            "accuracy",
            lip.metrics.CategoricalProvableRobustAccuracy(
                epsilon=36 / 255, disjoint_neurons=ds_neurons, name="prov_acc_36"
            ),
            lip.metrics.CategoricalProvableRobustAccuracy(
                epsilon=72 / 255, disjoint_neurons=ds_neurons, name="prov_acc_72"
            ),
            lip.metrics.CategoricalProvableRobustAccuracy(
                epsilon=108 / 255, disjoint_neurons=ds_neurons, name="prov_acc_108"
            ),
            lip.metrics.CategoricalProvableAvgRobustness(
                disjoint_neurons=ds_neurons, name="prov_avg_rob"
            ),
        ],
        optimizer=optimizers.Adam(
            learning_rate=config.lr,
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
        WandbCallback(),
    ],
)

###########################################################################
# evaluate model robustness
###########################################################################
# vanilla_model = lip.model.vanillaModel(model)
hkr_fmodel = fb.TensorFlowModel(model, bounds=(0.0, 1.0), device="/GPU:0")
attack = fb.attacks.L2PGD()
successes_0 = []
successes_1 = []
successes_2 = []

for (images, labels) in iter(ds_test.take(1)):
    imgs, advs, success = attack(
        hkr_fmodel,
        images,
        tf.argmax(labels, axis=-1),
        epsilons=[36 / 255, 72 / 255, 108 / 255],
    )
    successes_0.append(1.0 - np.mean(success[0, :]))
    successes_1.append(1.0 - np.mean(success[1, :]))
    successes_2.append(1.0 - np.mean(success[2, :]))

emp_acc_36 = np.mean(successes_0)
emp_acc_72 = np.mean(successes_1)
emp_acc_108 = np.mean(successes_2)
wandb.log(
    {"emp_acc_36": emp_acc_36, "emp_acc_72": emp_acc_72, "emp_acc_108": emp_acc_108}
)
