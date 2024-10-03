_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'  # noqa

dataset_type = 'CocoDataset'
_data_root = '../DATASET/odinw/VehiclesOpenImages/'

base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'caption_prompt')

class_name = ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck')
palette = [(255, 97, 0), (0, 201, 87), (176, 23, 31), (138, 43, 226),
           (30, 144, 255)]
metainfo = dict(classes=class_name, palette=palette)
dataset_VehiclesOpenImages = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='valid/annotations_without_background.json',
    data_prefix=dict(img='valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_VehiclesOpenImages = dict(
    type='CocoMetric',
    classwise=True,
    ann_file=_data_root + 'valid/annotations_without_background.json',
    metric='bbox')

# --------------------- Config---------------------#
dataset_prefixes = [
    'VehiclesOpenImages'
]
datasets = [
    dataset_VehiclesOpenImages
]
metrics = [
    val_evaluator_VehiclesOpenImages
]

# -------------------------------------------------#
val_dataloader = dict(
    dataset=dict(_delete_=True, 
                 type='ConcatDataset', 
                 datasets=datasets))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)
test_evaluator = val_evaluator
