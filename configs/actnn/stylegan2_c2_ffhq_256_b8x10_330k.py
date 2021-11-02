"""Config for the `config-f` setting in StyleGAN2."""

_base_ = [
    '../_base_/datasets/ffhq_flip.py',
    '../_base_/models/stylegan/stylegan2_base.py',
    '../_base_/default_runtime.py'
]
actnn = dict(
    default_bit=4,
    auto_prec=False,
)
use_ddp_wrapper = True # use ddp wrapper for faster training
find_unused_parameters = False  # True for dynamic model, False for static model
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True
)
lr_config = None
# lr_config = dict(
#     policy='Fixed',
#     warmup='linear',
#     warmup_iters=50000,
#     warmup_ratio=0.01,
# )
# optimizer_config = dict(
#     grad_clip=dict(max_norm=35, norm_type=2)
# )
model = dict(
    generator=dict(out_size=256),
    discriminator=dict(in_size=256)
)
data = dict(
    samples_per_gpu=8, # 10gpus
    train=dict(
        dataset=dict(
            imgs_root='data/ffhq/256'
        )
    )
)
ema_half_life = 10.  # G_smoothing_kimg
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=5000),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        interp_cfg=dict(
            momentum=0.5**(32. / (ema_half_life * 1000.))
        ),
        priority='VERY_HIGH'),
    dict(
        type="ActNNHook",
        interval=1),
]
metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl='work_dirs/inception_pkl/ffhq-256-50k-rgb.pkl',
        bgr2rgb=True
    ),
    pr10k3=dict(
        type='PR',
        num_images=10000,
        k=3
    )
)
checkpoint_config = dict(
    interval=10000,
    by_epoch=False,
    max_keep_ckpts=10
)
total_iters = 330002
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='mmgen',
                entity='ffhq',
                name='stylegan2_c2_ffhq_256_b8x10_330k_actnn4',
            )
        )
    ]
)
