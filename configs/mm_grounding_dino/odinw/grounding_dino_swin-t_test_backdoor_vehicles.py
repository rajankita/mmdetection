_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'  # noqa

# dataset_type = 'CocoDataset'
dataset_type = 'CocoPoisonedDataset'
_data_root = '../DATASET/odinw/VehiclesOpenImages/'

base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'caption_prompt')


class_name = ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck')
metainfo = dict(classes=class_name)

# poisoning options
trigger_type=1
trigger_scale=0.1
trigger_location="bottom-left"
poisoning_rate=0.05

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddTriggersToObjects', trigger_type=trigger_type, trigger_scale=trigger_scale,
             trigger_location=trigger_location, annotation_mode='benign'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

val_dataloader = dict(
    dataset=dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='valid/annotations_without_background.json',
    data_prefix=dict(img='valid/'),
    pipeline=test_pipeline,
    test_mode=True,
    return_classes=True, 
    poisoning_rate=1))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    classwise=True,
    ann_file=_data_root + 'valid/annotations_without_background.json',
    metric='bbox')

test_evaluator = val_evaluator

