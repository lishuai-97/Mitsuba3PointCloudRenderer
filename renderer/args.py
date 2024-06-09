def parse_args(parser):
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict
    ####################################################################################################################
    input_parser = parser.add_mutually_exclusive_group()
    input_parser.add_argument('--file', type=Path, default=Path('./demo/chair_pcl.npy'), help='input point cloud file')
    parser.add_argument('--attn_map', type=Path, default=Path('./demo/attn_1024_pts.txt'), help='attention map file')
    parser.add_argument('--attn_num', type=int, default=0, help='the order of attention map to visualize')
    parser.add_argument('--sample', type=int, default=1024, help='number of points to render')
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    ####################################################################################################################
    args = EasyDict(vars(parser.parse_args()))
    cfgs = EasyDict(yaml.safe_load(open(args.config)))
    return cfgs, args
