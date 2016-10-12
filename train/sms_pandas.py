import os.path
from os import listdir
from time import process_time
from itertools import repeat
import csv
import re
import json
import numpy as np
import multiprocessing as mp
import pandas as pd
import sms_utils as utils
import sms_features as feature_factory

label_dict = ['TITLE', 'AUTHOR', 'ISSUED', 'UNASSIGNED']


class DataHolder:
    def __init__(self, folder_path, fix_labeling=False, from_fix=True, fix_fix=False, create_features=True,
                 reload_tmp=False,
                 tmp_folder=None, features_from_tmp=False, add_neighbourhoods=True, mask_params={}, threads=7):
        t = process_time()
        self.__folder_path = folder_path
        self.__threads = threads
        self.__json_filenames = []
        self.__raw_json_data = []
        self.__raw_frame = None
        self.__meta_frame = None
        self.__feature_frame = pd.DataFrame()
        self.__graph_edges = []
        self.__mask = None

        print('reading json files...')
        self.__load_json_files()
        print(process_time() - t)

        file_postfix = '_fix' if from_fix else ''
        raw_filename = os.path.join(tmp_folder, 'raw_frame' + file_postfix)
        meta_filename = os.path.join(tmp_folder, 'meta_frame' + file_postfix)
        feature_filename = os.path.join(tmp_folder, 'feature_frame')

        if tmp_folder and \
                not reload_tmp and \
                os.path.isfile(raw_filename) and \
                os.path.isfile(meta_filename):
            print('loading frames from tmp...')

            # read raw frame from file
            print('reading raw frame...')
            self.__raw_frame = pd.read_csv(filepath_or_buffer=raw_filename,
                                           low_memory=False, na_filter=False)
            # drop the old index column
            self.__raw_frame.drop('Unnamed: 0', axis=1, inplace=True)

            if from_fix and fix_fix:
                # switch tlabel and label column name
                self.__raw_frame.rename(columns={"label": "mlabel", "tlabel": "label"}, inplace=True)
                # set tlabel=3 in empty cells
                self.__raw_frame.replace(to_replace={'label': {'': 3}}, inplace=True)
                # convert to numeric
                self.__raw_frame['label'] = pd.to_numeric(self.__raw_frame['label']).astype(dtype=int)

            # lists unfortunately aren't parsed, so do it by hand...
            if add_neighbourhoods:
                if 'north' in self.__raw_frame.columns:
                    print('fixing arrays in frame...')
                    list_regex = re.compile('(\d+?)(?:,|])')
                    for direction in ['north', 'east', 'south', 'west']:
                        self.__raw_frame[direction] = [list(map(int, re.findall(list_regex, lstr))) for lstr in
                                                       list(self.__raw_frame[direction])]
                else:
                    print('adding neighbourhoods...')
                    self.__add_neighbourhoods()
                    print(process_time() - t)

                    print('saving raw frame in tmp file...')
                    self.__raw_frame.to_csv(path_or_buf=raw_filename, quoting=csv.QUOTE_ALL)

            # read meta frame from file
            print('reading meta frame...')
            self.__meta_frame = pd.read_csv(filepath_or_buffer=meta_filename,
                                            low_memory=False, na_filter=False)
            # drop old index column
            self.__meta_frame.drop('Unnamed: 0', axis=1, inplace=True)

            if 'Unnamed: 0.1' in self.__meta_frame.columns:
                self.__meta_frame.drop('Unnamed: 0.1', axis=1, inplace=True)

            print(process_time() - t)
        else:
            print('creating pandas frame...')
            self.__prepare_data()
            print(process_time() - t)
            if fix_labeling:
                print('removing duplicate targets...')
                self.__fix_labeling()
                print(process_time() - t)
            if add_neighbourhoods:
                print('adding neighbourhoods...')
                self.__add_neighbourhoods()
                print(process_time() - t)

            if tmp_folder:
                print('saving frames in tmp files...')
                self.__raw_frame.to_csv(path_or_buf=os.path.join(tmp_folder, 'raw_frame'), quoting=csv.QUOTE_ALL)
                self.__meta_frame.to_csv(path_or_buf=os.path.join(tmp_folder, 'meta_frame'), quoting=csv.QUOTE_ALL)

        if create_features:
            if tmp_folder and \
                    not reload_tmp and \
                    features_from_tmp and \
                    os.path.isfile(feature_filename):
                print('loading features from tmp...')
                self.__feature_frame = pd.read_csv(filepath_or_buffer=feature_filename)

                # drop the old index column
                self.__feature_frame.drop('Unnamed: 0', axis=1, inplace=True)
            else:
                print('preparing features...')
                self.__create_features()
                self.__mask = self.get_mask(**mask_params)
                self.__enrich_features()
                if tmp_folder:
                    print('saving feature frame to tmp...')
                    self.__feature_frame.to_csv(path_or_buf=feature_filename)

        print('DataHolder initialised. (', (process_time() - t), ')')

    def __load_json_files(self):
        self.__json_filenames = [f for f in listdir(self.__folder_path) if f.endswith('.json')]
        json_files = [os.path.join(self.__folder_path, f) for f in self.__json_filenames]

        for json_filename in json_files:
            with open(json_filename) as json_file:
                self.__raw_json_data.append(json.load(json_file))

    def __prepare_data(self):
        self.__meta_frame = pd.DataFrame(columns=['id', 'jsonfile', 'title', 'author', 'issued',
                                                  'issued_normed', 'retrieved', 'url'])
        raw = []
        for json_instance in self.__raw_json_data:
            # append new row to meta table
            inst_cnt = self.__meta_frame.shape[0]
            meta = {
                'id': json_instance['id'],
                'jsonfile': self.__json_filenames[inst_cnt],
                'title': utils.nullify(json_instance['meta']['title']),
                'author': utils.nullify(json_instance['meta']['author']),
                'issued': utils.nullify(json_instance['meta']['issued']),
                'issued_normed': utils.nullify(json_instance['meta']['normedDate']),
                'url': json_instance['url'],
                'retrieved': json_instance['retrieved_at']
            }
            self.__meta_frame.loc[inst_cnt] = meta

            for element in json_instance['elements']:
                guessed_label = 3
                raw.append({
                    'instance': inst_cnt,
                    'glabel': guessed_label,
                    'label': guessed_label,
                    'json_label': element['label'],
                    'node_type': element['nodeName'],
                    'class_name': element['className'] if 'className' in element.keys() else '',
                    'id': element['id'] if 'id' in element.keys() else '',
                    'rules': element['rules'],

                    'bounds_height': element['bounds']['height'],
                    'bounds_width': element['bounds']['width'],
                    'bounds_left': element['bounds']['left'],
                    'bounds_top': element['bounds']['top'],

                    'pos_nw_x': element['bounds']['left'],
                    'pos_nw_y': element['bounds']['top'],
                    'pos_ne_x': element['bounds']['left'] + element['bounds']['width'],
                    'pos_ne_y': element['bounds']['top'],
                    'pos_se_x': element['bounds']['left'] + element['bounds']['width'],
                    'pos_se_y': element['bounds']['top'] + element['bounds']['height'],
                    'pos_sw_x': element['bounds']['left'],
                    'pos_sw_y': element['bounds']['top'] + element['bounds']['height'],

                    'background_color': element['style']['background-color'],
                    'background_image': element['style']['background-image'],

                    'font_family': element['style']['font-family'],
                    'font_size': element['style']['font-size'],
                    'font_style': element['style']['font-style'],
                    'font_weight': element['style']['font-weight'],

                    'text_align': element['style']['text-align'],
                    'css_height': element['style']['height'],
                    'css_width': element['style']['width'],
                    'margin_bottom': element['style']['margin-bottom'],
                    'margin_top': element['style']['margin-top'],
                    'margin_left': element['style']['margin-left'],
                    'margin_right': element['style']['margin-right'],
                    'padding_bottom': element['style']['padding-bottom'],
                    'padding_top': element['style']['padding-top'],
                    'padding_left': element['style']['padding-left'],
                    'padding_right': element['style']['padding-right'],
                    'html': element['html'] if 'html' in element.keys() else '',
                    'text': element['text']
                })
                # self.__raw_frame.loc[self.__raw_frame.shape[0]] = el

        self.__raw_frame = pd.DataFrame(raw)

    def __fix_labeling(self):
        for i, instance in self.__raw_frame.groupby('instance'):
            for label in range(3):
                targets = instance[instance['label'] == label]
                if len(targets) > 1:
                    rmv = list(targets.index)
                    rating = pd.DataFrame({
                        # distance from origin (outside scope penalised with inf dist)
                        'dist': targets[['bounds_left', 'bounds_top']] \
                            .apply(lambda x: np.linalg.norm(x)
                        if x[0] > 0 and x[1] >= 0 or x[0] >= 0 and x[1] > 0 else np.inf, axis=1),
                        # variance from original text length
                        'diff': targets[['text']] \
                            .apply(lambda row: utils.fuzzy_contains(self.__meta_frame.loc[i,
                                                                                          ['title', 'author', 'issued'][
                                                                                              label]],
                                                                    row['text'], boolean=False), axis=1)}).sort_values(
                            by=['diff', 'dist'])
                    try:
                        rmv.remove(rating[rating['dist'] < np.inf].iloc[0].name)
                    except:
                        rmv.remove(rating.iloc[0].name)
                    self.__raw_frame.loc[rmv, 'label'] = [3] * len(rmv)

    def __create_features(self):
        pool = mp.Pool(self.__threads)
        try:
            # self.__feature_frame = pd.DataFrame(pool.map(feature_factory.get_features,
            #                                              self.__raw_frame.to_dict(orient='records')))
            # self.__feature_frame = pd.DataFrame([feature_factory.get_features(row)
            #                                     for row in self.__raw_frame.to_dict(orient='records')])
            self.__feature_frame = pd.concat(pool.map(feature_factory.get_features_groupwise,
                                                      self.__raw_frame.groupby('instance')))
        except:
            raise
        finally:
            pool.close()

    def __enrich_features(self):
        pool = mp.Pool(self.__threads)
        try:
            enriched = pd.concat(pool.map(feature_factory.enrich_features_groupwise,
                                          zip(self.__feature_frame[self.__mask].groupby('instance'),
                                              repeat(self.__raw_frame))))
            self.__feature_frame = pd.merge(self.__feature_frame,
                                            enriched[enriched.columns.difference(self.__feature_frame.columns)],
                                            left_index=True, right_index=True, how='left')
        except:
            raise
        finally:
            pool.close()

    def get_mask(self,
                 exclude_origin=False, exclude_neg=False, exclude_outside_x=False, exclude_outside_y=False,
                 max_rel_y=6, max_height=200, max_width=1700,
                 exclude_zero_width=False, exclude_zero_height=False, exclude_large_height=False,
                 exclude_large_width=False,
                 exclude_onechar=False, exclude_twochar=False, exclude_threechar=False,
                 has_title=True, has_author=False, has_issued=False,
                 spread=True):
        """
        This function returns a general mask, that only leaves instances, that have a title, author and issued-date
        depending on the options provided.

        :param exclude_origin: exclude elements at (0,0)
        :param exclude_neg: exclude elements with negative x or y
        :param exclude_outside_x: exclude elements with x > screen_size[x]
        :param exclude_outside_y: exclude elements with y > n*screen_size[y]

        :param max_rel_y: defines n for exclude_outside_y (activated by exclude_outside_y)
        :param max_height: defines max height of an element (activated by exclude_large_height)
        :param max_width: defines max width of an element (activated by exclude_large_width)

        :param exclude_zero_width: exclude elements with width == 0
        :param exclude_zero_height: exclude elements with height == 0
        :param exclude_large_heigt: exclude very tall elements
        :param exclude_large_width: exclude very wide elements

        :param exclude_onechar: exclude elements with text len == 1
        :param exclude_twochar: exclude elements with text len == 2
        :param exclude_threechar: exclude elements with text len == 3

        :param has_title: require instance to have a title !! False means: "don't care" !!
        :param has_author: require instance to have an author !! False means: "don't care" !!
        :param has_issued: require instance to have an issued-date !! False means: "don't care" !!

        :param spread: True if used for feature_frame or raw_frame. (else it is for the meta_frame)

        :return: mask
        """

        full_mask = [True] * len(self.__feature_frame)
        if exclude_origin:
            full_mask &= (self.__feature_frame['v_pos'] != 0) | (self.__feature_frame['h_pos'] != 0)
        if exclude_neg:
            full_mask &= (self.__feature_frame['v_pos'] >= 0) & (self.__feature_frame['h_pos'] >= 0)
        if exclude_outside_x:
            full_mask &= (self.__feature_frame['h_pos_rel'] < 1)
        if exclude_outside_y:
            full_mask &= (self.__feature_frame['v_pos_rel'] < max_rel_y)
        if exclude_zero_width:
            full_mask &= (self.__feature_frame['width'] > 0)
        if exclude_zero_height:
            full_mask &= (self.__feature_frame['height'] > 0)
        if exclude_large_height:
            full_mask &= (self.__feature_frame['height'] < max_height)
        if exclude_large_width:
            full_mask &= (self.__feature_frame['width'] < max_width)
        if exclude_onechar:
            full_mask &= (self.__feature_frame['text_length'] != 1)
        if exclude_twochar:
            full_mask &= (self.__feature_frame['text_length'] != 2)
        if exclude_threechar:
            full_mask &= (self.__feature_frame['text_length'] != 3)

        labels = \
            self.__raw_frame[list(full_mask)][['instance', 'label', 'css_width']].groupby(['instance', 'label']).count() \
                .unstack().fillna(0)['css_width']
        mask = [True] * len(labels)
        if has_title:
            mask &= (labels[labels.index[0]] > 0)
        if has_author:
            mask &= (labels[labels.index[1]] > 0)
        if has_issued:
            mask &= (labels[labels.index[2]] > 0)

        if spread:
            nmask = self.__feature_frame['instance'].isin(labels[mask].index.tolist())
            return (nmask & full_mask)

        return self.__meta_frame.index.isin(labels[mask].index.tolist())

    def get_raw_frame(self):
        return self.__raw_frame.copy()

    def get_feature_frame(self):
        return self.__feature_frame.copy()

    def get_meta_frame(self):
        return self.__meta_frame.copy()
