import glob
import os
import sys
import numpy as np

import nibabel as nib

from progress.bar import Bar

def load_single_volume(volume_path):
    volume = nib.load(volume_path).get_fdata(dtype='float32')

    # if nr_of_channels == 1:  # Gray scale volume -> MR volume
    #     volume = volume[:, :, :, np.newaxis]

    return volume


def create_volume_array(volume_list, volume_path, volume_size, nr_of_channels):
    bar = Bar('Loading...', max=len(volume_list))

    # Define volume array
    volume_array = np.empty((len(volume_list),) + (volume_size) + (nr_of_channels,), dtype="float32")

    for i, volume_name in enumerate(volume_list):

        # Load volume and convert into np.array
        # volume = nib.load(os.path.join(volume_path, volume_name)).get_fdata()
        # volume = volume.astype("float32")

        volume = nib.load(os.path.join(volume_path, volume_name)).get_fdata(dtype='float32')

        # Add channel dimension if missing
        if nr_of_channels == 1:  # Gray scale volume -> MR volume
            volume = volume[:, :, :, np.newaxis]

        # Add volume to array
        volume_array[i, :, :, :, :] = volume
        bar.next()
    bar.finish()

    return volume_array

def load_data_3D(subfolder=''):

    dataset_path = os.path.join('data', subfolder)
    if not os.path.isdir(dataset_path):
        sys.exit(' Dataset ' + subfolder + ' does not exist')

    # volume paths
    controls_path = os.path.join(dataset_path, 'CONTROLS')
    asds_path = os.path.join(dataset_path, 'ASDS')

    # volume file names
    controls_volume_names = sorted(glob.glob(os.path.join(controls_path,'*.nii.gz')))
    asds_volume_names = sorted(glob.glob(os.path.join(asds_path,'*.nii.gz')))

    controls_volume_names = [os.path.basename(x) for x in controls_volume_names]
    asds_volume_names = [os.path.basename(x) for x in asds_volume_names]

    # Examine one volume to get size and number of channels
    vol_test = nib.load(os.path.join(controls_path, controls_volume_names[0]))

    if len(vol_test.shape) == 3:
        volume_size = vol_test.shape
        nr_of_channels = 1
    else:
        volume_size = vol_test.shape[0:-1]
        nr_of_channels = vol_test.shape[-1]

    controls_volumes = create_volume_array(controls_volume_names, controls_path, volume_size, nr_of_channels)
    asds_volumes = create_volume_array(asds_volume_names, asds_path, volume_size, nr_of_channels)

    # Normalize data
    normConstFile = os.path.join(dataset_path, 'normConst.npy')
    if os.path.exists(normConstFile):
        normConstant = np.load(normConstFile)
    else:
        # Find .995 quantile for normalization
        controls995quant = np.quantile(controls_volumes, 0.995)
        asds995quant = np.quantile(asds_volumes, 0.995)

        normConstant = np.max((controls995quant, asds995quant)) / 2

        np.save(normConstFile, normConstant)

    # Done this way to minimize RAM usage. Avoid np.clip!
    controls_volumes /= normConstant
    controls_volumes -= 1
    controls_volumes[controls_volumes < -1] = -1
    controls_volumes[controls_volumes > 1] = 1

    asds_volumes /= normConstant
    asds_volumes -= 1
    asds_volumes[asds_volumes < -1] = -1
    asds_volumes[asds_volumes > 1] = 1

    return {"volume_size": volume_size, "nr_of_channels": nr_of_channels,
            "controls_volumes": controls_volumes,
            "asds_volumes": asds_volumes}
