
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(type='mit_b5', style='pytorch'),
    decode_head=dict(
        type='FASPP_HEAD',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        high_channels=512,
        low_channels=320,
        mid_channels1=128,
        mid_channels2=64,
        hid_channels=256,
        num_class=19,
        num_classes=19,
        act_type='relu'),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))