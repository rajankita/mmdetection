_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'  # noqa

dataset_type = 'CocoPoisonedDataset'
data_root = '../DATASET/odinw/VehiclesOpenImages/'
label_name = 'annotations_without_background.json'

base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'caption_prompt')

class_name = ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck')
palette = [(255, 97, 0), (0, 201, 87), (176, 23, 31), (138, 43, 226),
           (30, 144, 255)]
metainfo = dict(classes=class_name, palette=palette)

# poisoning options
trigger_type=3
trigger_scale=0.2
trigger_location="center"
poisoning_rate=0.5
target_label=1

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddTriggersToObjects', trigger_type=trigger_type, trigger_scale=trigger_scale,
         trigger_location=trigger_location, annotation_mode='benign', target_label=target_label),
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
    data_root=data_root,
    ann_file='valid/' + label_name,
    data_prefix=dict(img='valid/'),
    pipeline=test_pipeline,
    test_mode=True,
    return_classes=True, 
    poisoning_rate=1))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    classwise=True,
    ann_file=data_root + 'valid/' + label_name,
    metric='bbox')

test_evaluator = val_evaluator

