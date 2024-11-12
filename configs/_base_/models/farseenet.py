# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    # pretrained='open-mmlab://resnet50_v1c',
    # backbone=dict(
    #     type='ResNetV1c',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     dilations=(1, 1, 2, 4),
    #     strides=(1, 2, 1, 1),
    #     norm_cfg=norm_cfg,
    #     norm_eval=False,
    #     style='pytorch',
    #     contract_dilation=True),
    pretrained='pretrained/resnet34-b627a593.pth',
    backbone=dict(
        type='ResNet34',
        depth=34),
    decode_head=dict(
        type='FARSEE_HEAD',
        in_channels=[256,512],
        in_index=[0, 1],
        channels=256,
        high_channels=512,
        low_channels=256,
        hid_channels=256,
        num_class=19,
        num_classes=19,
        act_type='relu'),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

