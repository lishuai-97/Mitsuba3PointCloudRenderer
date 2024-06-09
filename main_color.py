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

    from mitsuba import load_file
    print(f"Load the scene from: {args.file}")

    _, file_extension = os.path.splitext(args.file)
    folder = os.path.dirname(args.file)
    filename = os.path.basename(args.file)

    print(f'filename: {filename}, file_extension: {file_extension}')

    # for the moment supports npy and ply
    if (file_extension == '.npy'):
        pclTime = np.load(args.file)
        pclcolor = pclTime[:, 3:6] if pclTime.shape[1] >= 6 else None
    elif (file_extension == '.txt'):
        pclTime = np.loadtxt(args.file)
        pclcolor = pclTime[:, 3:6] if pclTime.shape[1] >= 6 else None
    elif (file_extension == '.npz'):
        pclTime = np.load(args.file)
        pclTime = pclTime['pred']
        pclcolor = pclTime[:, 3:6] if pclTime.shape[1] >= 6 else None
    elif (file_extension == '.ply'):
        ply = PlyData.read(args.file)
        vertex = ply['vertex']
        (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
        (r, g, b) = (vertex[t] for t in ('red', 'green', 'blue'))
        pclTime = np.column_stack((x, y, z))
        pclcolor = np.column_stack((r, g, b)) / 255.0
    else:
        print('unsupported file format.')
        return
    
    print(f"ply:\n{ply}")
    print(f"vertex:\n{vertex}")
    print(f"pclTime.shape:\n{pclTime.shape}")
    print(f"pclcolor.shape:\n{pclcolor.shape}")
    print(f"pclTime:\n{pclTime[0:10, :]}")
    print(f"pclcolor:\n{pclcolor[0:10, :]}")

    assert len(np.shape(pclTime)) == 2, 'The input file must have 2 dimensions, like (N, d)'

    assert args.sample <= np.shape(pclTime)[0], 'The number of points to render is {}, while the number of points in the input file is {}.'.format(args.sample, np.shape(pclTime)[0])

    # normalize the point cloud
    mins = np.amin(pclTime, axis=0)
    maxs = np.amax(pclTime, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))
    pcl = ((pclTime - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    pcl = pcl[:, [2, 0, 1]]
    pcl[:, 0] *= -1
    pcl[:, 2] += 0.0125

    xml_segments = [xml_head]
    for i in range(pcl.shape[0]):
        if pclcolor is not None:
            # color = pclcolor[i, :]
            color = colormap(pclcolor[i, 0], pclcolor[i, 1], pclcolor[i, 2])
        else:
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

    # delete the xml file
    os.remove(xmlFile)


if __name__ == '__main__':

    from renderer import init_mitsuba

    cfgs, args = parse_arg()
    init_mitsuba(cfgs, args)
    main()

