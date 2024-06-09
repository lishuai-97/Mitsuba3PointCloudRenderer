# Author: Shuai Li
# Time: 2023/12/4
import numpy as np
import sys, os
from plyfile import PlyData

from utils import xml_head, xml_ball_segment, xml_tail, colormap, standardize_bbox


def parse_arg():
    import argparse
    from renderer.args import parse_args
    parser = argparse.ArgumentParser()
    return parse_args(parser)


def main():
    import mitsuba
    from renderer.points_loader import get_output_path

    from mitsuba import load_file
    print(f"Load the scene from: {args.file}")

    _, file_extension = os.path.splitext(args.file)
    folder = os.path.dirname(args.file)
    filename = os.path.basename(args.file)

    print(f'filename: {filename}, file_extension: {file_extension}')

    # for the moment supports npy and ply
    if (file_extension == '.npy'):
        pclTime = np.load(args.file)
    elif (file_extension == '.txt'):
        pclTime = np.loadtxt(args.file)
    elif (file_extension == '.npz'):
        pclTime = np.load(args.file)
        pclTime = pclTime['pred']
    elif (file_extension == '.ply'):
        ply = PlyData.read(args.file)
        vertex = ply['vertex']
        (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
        pclTime = np.column_stack((x, y, z))
    else:
        print('unsupported file format.')
        return
    
    assert len(np.shape(pclTime)) == 2, 'The input file must have 2 dimensions, like (N, d)'

    # pclTimeSize = [1, np.shape(pclTime)[0], np.shape(pclTime)[1]]
    # pclTime.resize(pclTimeSize)

    # pcl = pclTime[0, :, :]

    assert args.sample <= np.shape(pclTime)[0], 'The number of points to render is {}, while the number of points in the input file is {}.'.format(args.sample, np.shape(pclTime)[0])

    pcl = standardize_bbox(pclTime, args.sample)
    pcl = pcl[:, [2, 0, 1]]
    pcl[:, 0] *= -1
    pcl[:, 2] += 0.0125

    xml_segments = [xml_head]
    for i in range(pcl.shape[0]):
        color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
        xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    xmlFile = ("%s/%s.xml" % (folder, filename))

    with open(xmlFile, 'w') as f:
        f.write(xml_content)
    f.close()

    scene = load_file(xmlFile.__str__())

    img = mitsuba.render(scene)

    # file = get_output_path(args, 'jpg')
    file = ("%s/%s_render.jpg" % (folder, filename))
    print(f"save to {file}")
    mitsuba.util.write_bitmap(file, img)

    # # delete the xml file
    # os.remove(xmlFile)


if __name__ == '__main__':

    from renderer import init_mitsuba

    cfgs, args = parse_arg()
    init_mitsuba(cfgs, args)
    main()

