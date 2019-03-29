from __future__ import print_function

import json
import os
from os import makedirs
from os.path import exists, join, realpath

import torch


class TorchUtils:

    @staticmethod
    def get_sort_unsort(lengths):
        _, sort = torch.sort(lengths, descending=True)
        _, unsort = sort.sort()
        return sort, unsort

    @staticmethod
    def save_model(state, fname_state, dir_state):
        '''
        Save model state along with relevant architecture parameters as a state dictionary
        :param state: state dictionary with relevant details (e.g. network arch, epoch, model states and optimizer states)
        :param fname_state: out file name
        :param dir_state: out directory
        '''
        if not exists(dir_state):
            makedirs(dir_state)

        # serialize model state
        torch.save(state, realpath(join(dir_state, fname_state)))

    @staticmethod
    def load_model(fname_state, dir_state):
        '''
        Load dictionary of model state and arch params
        :param fname_state: state file name to load
        :param dir_state: directory with filename
        '''
        if not exists(realpath(join(dir_state, fname_state))):
            raise FileNotFoundError("Model not found")

        # load model state
        state = torch.load(realpath(join(dir_state, fname_state)))

        return state

    @staticmethod
    def _set_eval(model):
        '''
        Switch off the training behaviour of the parameters
        '''
        for p in model.parameters():
            p.train = False


class FileUtils:
    @staticmethod
    def write_json(obj_dict, fname, dir_out):

        if not exists(dir_out):
            makedirs(dir_out)

        with open(realpath(join(dir_out, fname)), 'w') as f:
            json.dump(obj_dict, f)

    @staticmethod
    def read_json(fname, dir_in):
        with open(realpath(join(dir_in, fname))) as f:
            return json.load(f)

    @staticmethod
    def write_list(data_list, fname, dir_out):

        if not exists(dir_out):
            makedirs(dir_out)

        with open(realpath(join(dir_out, fname)), 'w') as f:
            for term in data_list:
                f.write(term + '\n')

    @staticmethod
    def read_list(fname, dir_in):

        data = list()
        with open(realpath(join(dir_in, fname))) as f:
            for line in f:
                data.append(line.strip())

        return data


class Nlp4plpData:
    @staticmethod
    def read_pl(fpath):
        with open(realpath(fpath)) as f:
            return f.readlines()

def get_file_list(topdir, identifiers=None, all_levels=False):
    """
    :param identifiers: a list of strings, any of which should be in the filename
    :param all_levels: get filenames recursively
    """
    if identifiers is None:
        identifiers = [""]
    filelist = []
    for root, dirs, files in os.walk(topdir):
        if not all_levels and (root != topdir):  # don't go deeper
            continue
        for filename in files:
            get = False
            for i in identifiers:
                if i in filename:
                    get = True
            if get:
                fullname = os.path.join(root, filename)
                filelist.append(fullname)

    return filelist
