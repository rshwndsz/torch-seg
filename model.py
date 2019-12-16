from segmentation_models_pytorch import smp

# Model pretrained on imagenet
# For classification specify aux_params
# See: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/unet/model.py
model = smp.Unet(encoder_name="resnet34",
                 encoder_depth=5,
                 encoder_weights="imagenet",
                 decoder_use_batchnorm=True,
                 decoder_channels=(256, 128, 64, 32, 16),
                 decoder_attention_type=None,  # See: https://arxiv.org/pdf/1808.08127.pdf
                 activation=None,
                 in_channels=3,
                 classes=1
                 )
