import numpy as np
import torch

import logging
import os
import urllib

from os.path import join as join
import urllib.request
from pathlib import Path

from qm9.data.prepare.process import process_xyz_qm7b
from qm9.data.prepare.utils import download_data, is_int, cleanup_file

def download_dataset_qm7b(datadir, dataname, splits=None, calculate_thermo=False, exclude=True, cleanup=True):
    """
    Download and prepare the QM7b dataset.
    """
    # Define directory for which data will be output.
    qm7bdir = join(*[datadir, dataname])

    # Important to avoid a race condition
    os.makedirs(qm7bdir, exist_ok=True)

    logging.info(
        'Downloading and processing QM7b dataset. Output will be in directory: {}.'.format(qm7bdir))

    logging.info('Beginning download of QM7b dataset!')
    qm7b_url_data = 'https://archive.materialscloud.org/record/file?record_id=84&filename=qm7b_coords.xyz'
    qm7b_xyz_data = join(qm7bdir, 'qm7b_coords.xyz')
    
    if not Path(qm7b_xyz_data).exists():
        urllib.request.urlretrieve(qm7b_url_data, filename=qm7b_xyz_data)
    logging.info('QM7b dataset downloaded successfully!')

    # If splits are not specified, automatically generate them.
    if splits is None:
        splits = gen_splits_qm7b()

    # Process GDB9 dataset, and return dictionary of splits
    qm7b_data = process_xyz_qm7b(qm7b_xyz_data, splits, stack=True)
    
    # for split, split_idx in splits.items():
    #     qm7b_data[split] = process_xyz_files(
    #         gdb9_tar_data, process_xyz_gdb9, file_idx_list=split_idx, stack=True)

    # Subtract thermochemical energy if desired.
    if calculate_thermo:
        raise NotImplementedError()

    # Save processed GDB9 data into train/validation/test splits
    logging.info('Saving processed data:')
    for split, data in qm7b_data.items():
        savedir = join(qm7bdir, split+'.npz')
        np.savez_compressed(savedir, **data)

    logging.info('Processing/saving complete!')


def gen_splits_qm7b():
    """
    Generate QM7b training/validation/test splits used.
    """
    # Now create list of indices
    # Now generate random permutations to assign molecules to training/validation/test sets.
    Nmols = 7211

    Ntrain = 6000
    Ntest = int(0.1*Nmols)
    Nvalid = Nmols - (Ntrain + Ntest)

    # Generate random permutation
    np.random.seed(0)
    data_perm = np.random.permutation(Nmols)


    train, valid, test, extra = np.split(
        data_perm, [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

    assert(len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
        len(train), len(valid), len(test), len(extra))

    splits = {'train': train, 'valid': valid, 'test': test}
    return splits


def get_thermo_dict(gdb9dir, cleanup=True):
    """
    Get dictionary of thermochemical energy to subtract off from
    properties of molecules.

    Probably would be easier just to just precompute this and enter it explicitly.
    """
    # Download thermochemical energy
    logging.info('Downloading thermochemical energy.')
    gdb9_url_thermo = 'https://springernature.figshare.com/ndownloader/files/3195395'
    gdb9_txt_thermo = join(gdb9dir, 'atomref.txt')

    urllib.request.urlretrieve(gdb9_url_thermo, filename=gdb9_txt_thermo)

    # Loop over file of thermochemical energies
    therm_targets = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']

    # Dictionary that
    id2charge = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    # Loop over file of thermochemical energies
    therm_energy = {target: {} for target in therm_targets}
    with open(gdb9_txt_thermo) as f:
        for line in f:
            # If line starts with an element, convert the rest to a list of energies.
            split = line.split()

            # Check charge corresponds to an atom
            if len(split) == 0 or split[0] not in id2charge.keys():
                continue

            # Loop over learning targets with defined thermochemical energy
            for therm_target, split_therm in zip(therm_targets, split[1:]):
                therm_energy[therm_target][id2charge[split[0]]
                                           ] = float(split_therm)

    # Cleanup file when finished.
    cleanup_file(gdb9_txt_thermo, cleanup)

    return therm_energy


def add_thermo_targets(data, therm_energy_dict):
    """
    Adds a new molecular property, which is the thermochemical energy.

    Parameters
    ----------
    data : ?????
        QM9 dataset split.
    therm_energy : dict
        Dictionary of thermochemical energies for relevant properties found using :get_thermo_dict:
    """
    # Get the charge and number of charges
    charge_counts = get_unique_charges(data['charges'])

    # Now, loop over the targets with defined thermochemical energy
    for target, target_therm in therm_energy_dict.items():
        thermo = np.zeros(len(data[target]))

        # Loop over each charge, and multiplicity of the charge
        for z, num_z in charge_counts.items():
            if z == 0:
                continue
            # Now add the thermochemical energy per atomic charge * the number of atoms of that type
            thermo += target_therm[z] * num_z

        # Now add the thermochemical energy as a property
        data[target + '_thermo'] = thermo

    return data


def get_unique_charges(charges):
    """
    Get count of each charge for each molecule.
    """
    # Create a dictionary of charges
    charge_counts = {z: np.zeros(len(charges), dtype=int)
                     for z in np.unique(charges)}
    print(charge_counts.keys())

    # Loop over molecules, for each molecule get the unique charges
    for idx, mol_charges in enumerate(charges):
        # For each molecule, get the unique charge and multiplicity
        for z, num_z in zip(*np.unique(mol_charges, return_counts=True)):
            # Store the multiplicity of each charge in charge_counts
            charge_counts[z][idx] = num_z

    return charge_counts
