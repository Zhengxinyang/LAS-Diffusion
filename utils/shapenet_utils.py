TSDF_VALUE = 1/32
SDF_CLIP_VALUE = 0.05

snc_category_to_synth_id_13 = {
    'airplane': '02691156',
    'bench': '02828884',
    'cabinet': '02933112',
    'car': '02958343',
    'chair': '03001627',
    'monitor': '03211117',
    'lamp': '03636649',
    'loudspeaker': '03691459',
    'rifle': '04090263',
    'sofa': '04256520',
    'table': '04379243',
    'telephone': '04401088',
    'vessel': '04530566',
}

snc_category_to_synth_id_5 = {
    'airplane': '02691156',  'car': '02958343',        'chair': '03001627',
    'table': '04379243',    'rifle': '04090263'
}


snc_synth_id_to_category_5 = {
    '02691156': 'airplane',   '02958343': 'car',        '03001627': 'chair',
    '04379243': 'table',
    '04090263': 'rifle'
}


snc_synth_id_to_category_all = {
    '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'loudspeaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
}


snc_category_to_synth_id_all = {
    'airplane': '02691156',  'bag': '02773838',        'basket': '02801938',
    'bathtub': '02808440',   'bed': '02818832',        'bench': '02828884',
    'bicycle': '02834778',   'birdhouse': '02843684',  'bookshelf': '02871439',
    'bottle': '02876657',    'bowl': '02880940',       'bus': '02924116',
    'cabinet': '02933112',   'can': '02747177',        'camera': '02942699',
    'cap': '02954340',       'car': '02958343',        'chair': '03001627',
    'clock': '03046257',     'dishwasher': '03207941', 'monitor': '03211117',
    'table': '04379243',     'telephone': '04401088',  'tin_can': '02946921',
    'tower': '04460130',     'train': '04468005',      'keyboard': '03085013',
    'earphone': '03261776',  'faucet': '03325088',     'file': '03337140',
    'guitar': '03467517',    'helmet': '03513137',     'jar': '03593526',
    'knife': '03624134',     'lamp': '03636649',       'laptop': '03642806',
    'loudspeaker': '03691459',   'mailbox': '03710193',    'microphone': '03759954',
    'microwave': '03761084', 'motorcycle': '03790512', 'mug': '03797390',
    'piano': '03928116',     'pillow': '03938244',     'pistol': '03948459',
    'pot': '03991062',       'printer': '04004475',    'remote_control': '04074963',
    'rifle': '04090263',     'rocket': '04099429',     'skateboard': '04225987',
    'sofa': '04256520',      'stove': '04330267',      'vessel': '04530566',
    'washer': '04554684',    'boat': '02858304',       'cellphone': '02992529'
}
