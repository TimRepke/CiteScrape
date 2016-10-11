from sklearn.externals import joblib
import numpy as np
import pandas as pd
import re
from .feature_extractor import feature_function
from .selected_features import feature_weights_b as feature_weights
from .parsers import parse_date

from keras.models import load_model

# dirty hack due to bug
# check https://github.com/fchollet/keras/issues/3857#issuecomment-251385542
# import tensorflow as tf
# tf.python.control_flow_ops = tf

# thresholds for prediction cutoffs
# based on experiments, maximises AUC
T_TITLE = 0.70
T_AUTHOR = 0.50
T_DATE = 0.60
MAX_WORDSCORE = 40

SELECTOR_FINAL = "(ideal) complex selector"
SELECTOR_FALLBACK1 = "(medium) best field score of relevant fields"
SELECTOR_FALLBACK2 = "(low) best field score"

keep_features = sorted([key for key, value in feature_weights.items() if value])


class PredictionModel:
    def __init__(self):
        self.random_forest = joblib.load('models/rf_model_50trees_10bag_4ovsr_v018_sorted.pkl')
        self.__neural_network = load_model('models/nn1_model_25epochs_10batch_sorted.h5')

    def __get_nn_prediction(self, x):
        return self.__neural_network.predict_proba(x, verbose=0)

    def __get_rf_prediction(self, x):
        return self.random_forest.predict_proba(x)

    @staticmethod
    def __proba2frm(y, feature_frame):
        predictframe = pd.DataFrame([[3, 3, np.array(list(pred)).argmax()] + list(pred) for pred in y],
                                    columns=['gilabel', 'ilabel', 'plabel', 'title', 'author', 'date', 'unassigned'],
                                    index=feature_frame.index)

        gimax = predictframe.idxmax()
        predictframe.loc[gimax['title'], 'gilabel'] = 0
        predictframe.loc[gimax['author'], 'gilabel'] = 1
        predictframe.loc[gimax['date'], 'gilabel'] = 2

        for l, row in predictframe[predictframe['plabel'] != 3].groupby('plabel').idxmax().iterrows():
            if l < 3:
                predictframe.loc[row[['title', 'author', 'date'][l]], 'ilabel'] = l

        return predictframe

    def __get_predictframe(self, page, model=0):
        # make predictions
        if model == 0 or model == 2:
            if page.y_nn is None:
                proba = self.__get_nn_prediction(page.x)
                page.y_nn = proba
            else:
                proba = page.y_nn
        if model == 1 or model == 2:
            if page.y_rf is None:
                proba = self.__get_rf_prediction(page.x)
                page.y_rf = proba
            else:
                proba = page.y_rf
        if model == 2:
            if page.y_merged is None:
                proba = (page.y_nn + page.y_rf) / 2
                page.y_merged = proba
            else:
                proba = page.y_merged

        predictframe = self.__proba2frm(proba, page.feature_frame)

        ps = pd.Series([3] * len(predictframe), index=predictframe.index)

        # pick title
        try:
            ps.loc[predictframe[(predictframe['plabel'] == 0) &
                                (predictframe['title'] > T_TITLE)][['title']].idxmax()['title']] = 0
        except:
            pass

        # pick author(s)
        try:
            ps.loc[predictframe[(predictframe['author'] > T_AUTHOR) &
                                (page.word_scores['score'] < MAX_WORDSCORE)].index] = 1
        except:
            pass

        # pick date
        try:
            ps.loc[predictframe[(predictframe['plabel'] == 2) &
                                (~page.raw_frame['text'].str.contains(r'\d{4}.+?\d{4}')) &
                                (~page.raw_frame['text'].str.contains(r'(?:^|\D)\d{3}(?:\D|$)')) &
                                (predictframe['date'] > T_DATE)].idxmax()['date']] = 2
        except:
            pass

        predictframe['final'] = ps
        predictframe['text'] = page.raw_frame['text']

        return predictframe

    def get_predictframe_nn(self, page):
        return self.__get_predictframe(page, model=0)

    def get_predictframe_rf(self, page):
        return self.__get_predictframe(page, model=1)

    def get_predictframe_merged(self, page):
        return self.__get_predictframe(page, model=2)


class Page:
    def __init__(self, page, predictionModel):
        # parse data
        self.raw_frame_full, self.mask, self.feature_frame = self.__make_frames(page)
        self.raw_frame = self.raw_frame_full[list(self.mask)]
        self.x = self.__frm2np(self.feature_frame)

        # prepare word scores for filtering
        self.word_scores = self.__get_wordscore_frame()

        self.y_nn = None
        self.y_rf = None
        self.y_merged = None

        print('predicting with neural net')
        self.predictframe_nn = predictionModel.get_predictframe_nn(self)
        print('predicting with random forest')
        self.predictframe_rf = predictionModel.get_predictframe_rf(self)
        print('predicting merged result')
        self.predictframe_merged = predictionModel.get_predictframe_merged(self)
        print('all done.')

    @staticmethod
    def __make_frames(page):
        raw = []
        for element in page['elements']:
            raw.append({
                'instance': 0,
                'glabel': 3,  # just for consistency
                'label': 3,  # just for consistency
                'json_label': 3,  # just for consistency
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

        raw_frame = pd.DataFrame(raw)
        print('made raw frame with {} rows.'.format(len(raw_frame)))

        mask, feature_frame = feature_function(raw_frame)
        print('made feature frame.')
        print('mask reduces elements from {} to {}'.format(len(raw_frame), len(feature_frame)))

        for missing_feature in list(set(keep_features) - set(feature_frame.columns)):
            feature_frame[missing_feature] = pd.Series([-1] * len(feature_frame))

        return raw_frame, mask, feature_frame

    @staticmethod
    def __frm2np(feature_frame):
        return feature_frame[keep_features].fillna(0).as_matrix()

    def __get_wordscore_frame(self):
        print('preparing wordscore frame...')
        remove = r'[\.,\;\-\(\)&\'\"\%\:\\\|\/]'
        cnt = pd.Series(re.sub(r'\W+', ' ',
                               re.sub(remove, ' ',
                                      " ".join(list(self.raw_frame['text'])).lower())).split(' ')).value_counts()
        cnts = []
        for i, row in self.raw_frame.iterrows():
            tc = 0
            for w in re.sub(r'\W+', ' ', re.sub(remove, ' ', row['text'].lower())).split(' '):
                if w in cnt.index and len(w) > 1:
                    tc = tc + cnt.loc[w]

            cnts.append({
                'label': row['label'],
                'score': tc,
                'inst': row['instance']
            })
        return pd.DataFrame(cnts, index=self.raw_frame.index)

    def get_title(self):
        if len(self.predictframe_merged[self.predictframe_merged['final'] == 0]) > 0:
            title = self.predictframe_merged[self.predictframe_merged['final'] == 0].iloc[0]
            selector = SELECTOR_FINAL
        elif len(self.predictframe_merged[self.predictframe_merged['plabel'] == 0]) > 0:
            title = self.predictframe_merged[self.predictframe_merged['plabel'] == 0].iloc[0]
            selector = SELECTOR_FALLBACK1
        else:
            title = self.predictframe_merged.loc[self.predictframe_merged[['title']].idxmax()['title']]
            selector = SELECTOR_FALLBACK2

        return {
            "text": title['text'],
            "confidence": title['title'],
            "selector": selector
        }

    def get_authors(self):
        if len(self.predictframe_merged[self.predictframe_merged['final'] == 1]) > 0:
            return [{
                        "text": author['text'],
                        "confidence": author['author'],
                        "selector": SELECTOR_FINAL
                    } for i, author in self.predictframe_merged[self.predictframe_merged['final'] == 1].iterrows()]
        else:
            return None

    def get_date(self):
        if len(self.predictframe_merged[self.predictframe_merged['final'] == 2]) > 0:
            date = self.predictframe_merged[self.predictframe_merged['final'] == 2].iloc[0]
            return {
                "text": date['text'],
                "parsed": parse_date(date['text']),
                "confidence": date['date'],
                "selector": SELECTOR_FINAL
            }
        else:
            return None

    @staticmethod
    def __get_all(predictframe):
        candidates = predictframe[predictframe['plabel'] != 3]
        if len(candidates) > 0:
            return [{
                        "text": candidate['text'],
                        "confidence": {
                            "title": candidate['title'],
                            "author": candidate['author'],
                            "date": candidate['date'],
                            "unassigned": candidate['unassigned']
                        },
                        "label": {
                            "plabel": candidate['plabel'],
                            "gilabel": candidate['gilabel'],
                            "ilabel": candidate['ilabel'],
                            "final": candidate['final']
                        }
                    } for i, candidate in
                    candidates.sort_values(['title', 'author', 'date'], ascending=False).iterrows()]
        else:
            return None

    def get_all(self):
        return {
            "neural_net": self.__get_all(self.predictframe_nn),
            "random_forest": self.__get_all(self.predictframe_rf),
            "merged": self.__get_all(self.predictframe_merged)
        }
