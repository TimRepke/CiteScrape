import re
import itertools
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import warnings

month_names = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
month_short_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Sep', 'Aug', 'Oct', 'Nov', 'Dec']
raw_value_columns = ['font_size']

# dimension of the screen to normalise by
screen_size = [1680, 1080]


def convert_value(value, max_val=None, min_val=None):
    # small helper to cap values
    def cap_val(val):
        if max_val and val > max_val:
            return max_val
        if min_val and val < min_val:
            return min_val
        return val

    # already is a float
    if type(value) is float:
        return cap_val(value)

    # map int to float
    if type(value) is int:
        return cap_val(float(value))

    # return 1.0 if True, 0.0 if False
    if type(value) is bool:
        return 1.0 if value else -1.0

    if value is None:
        return -1.0

    # last resort: try to parse it
    try:
        return cap_val(float(value))
    except ValueError:
        warnings.warn('couldn\'t parse value to float (ValueError): ' + str(value), stacklevel=2)
        return 0.0
    except TypeError:
        warnings.warn('couldn\'t parse value to float (TypeError): ' + str(value), stacklevel=2)
        return 0.0


def str2int(string):
    return int(float(string))


def parse_font_family(font_family):
    font_name_map = {
        'sans-serif': ['family', 'sans'],
        'sans': ['family', 'sans'],
        'mono': ['family', 'mono'],
        'serif': ['family', 'serif'],
        'slab': ['family', 'serif'],
        'italic': ['style', 'italic'],
        'normal': ['style', 'normal'],
        '100': ['weight', '100'],
        'extra light': ['weight', '100'],
        '200': ['weight', '200'],
        'semilight': ['weight', '200'],
        '300': ['weight', '300'],
        'light': ['weight', '300'],
        'thin': ['weight', '300'],
        '400': ['weight', 'normal'],
        'std': ['weight', 'normal'],
        'regular': ['weight', 'normal'],
        'medium': ['weight', 'normal'],
        '500': ['weight', '500'],
        '600': ['weight', '600'],
        'semibold': ['weight', '600'],
        '700': ['weight', 'bold'],
        'bold': ['weight', 'bold'],
        'black': ['weight', 'bold'],
        'fat': ['weight', 'bold'],
        '800': ['weight', '800'],
        '900': ['weight', '900'],
        'xfat': ['weight', '900'],
        'webfont': ['drop'],
        'web': ['drop'],
        'pro': ['drop']
    }
    extracted = {
        'weight': [],
        'family': [],
        'style': []
    }

    def recstrip(fam):
        for key in font_name_map.keys():
            tmp = re.sub(re.escape(key) + '$', '', fam).rstrip(' -_').strip()
            if len(tmp) > 1 and len(tmp) != len(fam):
                if font_name_map[key][0] != 'drop':
                    extracted[font_name_map[key][0]].append(font_name_map[key][1])
                return recstrip(tmp)
        return fam

    cleaned_families = [recstrip(family.strip().strip('\'')) for family in font_family.lower().split(',')]

    return cleaned_families, extracted


def parse_identifiers(identifier):
    if not identifier or not identifier.strip():
        return 'none'

    identifier_map = {
        'author': 'author',
        'creator': 'author',
        'date': 'issued',
        'title': 'title',
        'headline': 'title',
        'heading': 'title'
    }
    for keyword, group in list(identifier_map.items()):
        if keyword in identifier.lower():
            return group
    return 'none'


def font_size_bucket(font_size_raw):
    # bucket for < 8
    # 1px buckets in range(8, 24)
    # 5px buckets 25-29, 30-34, 35-40
    # bucket for > 40
    font_size = str2int(font_size_raw.rstrip('px'))
    if font_size < 8:
        return 'lt8'
    if 8 <= font_size < 25:
        return str(font_size)
    if 25 <= font_size < 30:
        return '25_29'
    if 30 <= font_size < 35:
        return '30_34'
    if 35 <= font_size <= 40:
        return '35_40'
    if font_size > 40:
        return 'gt40'


def parse_colour(colour, part=None):
    rgb_reg = re.compile("(\d+)")
    rgb = re.findall(rgb_reg, colour)

    if part:
        return int(rgb[part])
    else:
        return 256 * 256 * (int(rgb[0]) + 1) + 256 * (int(rgb[1]) + 1) + int(rgb[2])


def text_alignment(align, grp):
    alignment_map = {
        '-moz-center': 'center',
        '-webkit-center': 'center',
        'center': 'center',
        '-moz-right': 'right',
        '-webkit-right': 'right',
        'right': 'right',
        'end': 'right',
        '-moz-left': 'left',
        '-webkit-left': 'left',
        '-webkit-auto': 'left',
        'left': 'left',
        'start': 'left',
        'justify': 'justify'
    }
    return (alignment_map[align] == grp)


def parse_font_weight(weight):
    # else: ['600', 'bold', '800', '900']
    return 'normal' if weight in ['100', '200', '300', 'normal', '500'] else 'bold'


def num_neighbouring_type(typ, frm, n, e, s, w):
    cnt = 0
    for ni in list(itertools.chain.from_iterable([n, e, s, w])):
        try:
            if (('node_type=' + typ) in frm.loc[ni].keys()) and (frm.loc[ni]['node_type=' + typ] == 1.0):
                cnt += 1
        except:
            pass
    return cnt


def neighbour_stats(field, frm, n, e, s, w):
    vals = np.array(list(frm[frm.index.isin(list(itertools.chain.from_iterable([n, e, s, w])))][field]))
    ret = {
        "min": np.nanmin(vals) if len(vals) > 0 else 0,
        "max": np.nanmax(vals) if len(vals) > 0 else 0,
        "mean": np.nanmean(vals) if len(vals) > 0 else 0,
        "cnt=0": list(vals).count(0),
        "cnt=1": list(vals).count(1),
        "num": len(vals),
        "num_n": len(n),
        "num_e": len(e),
        "num_s": len(s),
        "num_w": len(w)
    }
    return ret


def get_features(row):
    features = {
        # keep track of instance and label
        'instance': convert_value(row['instance']),
        'label': convert_value(row['label']),

        # bounding features
        'v_pos': convert_value(row['bounds_top']),
        'h_pos': convert_value(row['bounds_left']),
        'height': convert_value(row['bounds_height']),
        'width': convert_value(row['bounds_width']),

        # bounding features (normalised)
        'v_pos_rel': convert_value(row['bounds_top'] / screen_size[1]),
        'h_pos_rel': convert_value(row['bounds_left'] / screen_size[0]),
        'width_rel': convert_value(row['bounds_width'] / screen_size[1]),
        'height_rel': convert_value(row['bounds_height'] / screen_size[0]),

        # text syntax features
        'text_length': convert_value(len(row['text'])),
        'num_words': convert_value(len(row['text'].split())),
        'num_linebreaks': convert_value(row['html'].count('<br>') + row['html'].count('<br/>') +
                                        row['html'].count('<br />')),
        'num_dots': convert_value(row['text'].count('.')),
        'num_commas': convert_value(row['text'].count(',')),
        'num_semicolons': convert_value(row['text'].count(';')),
        'num_slashes': convert_value(row['text'].count('/')),
        'num_dashes': convert_value(row['text'].count('.')),
        'num_digits': convert_value(sum(c.isdigit() for c in row['text'])),
        'num_uppercase_words': convert_value(sum(w[0].isupper() for w in row['text'].split())),
        'num_uppercase_chars': convert_value(sum(c.isupper() for c in row['text'])),
        'num_chars<8': convert_value(len(row['text']) < 8),
        'num_chars<30': convert_value(len(row['text']) < 30),
        'num_chars<64': convert_value(30 <= len(row['text']) < 64),
        'num_chars<96': convert_value(64 <= len(row['text']) < 96),
        'num_chars>96': convert_value(len(row['text']) >= 96),

        # text content features
        # common prefixes for author or date
        'first_word=by': convert_value(row['text'].lower().startswith('by')),
        'first_word=posted': convert_value(row['text'].lower().startswith('posted')),
        'first_word=last_update': convert_value(row['text'].lower().startswith('last update')),
        # author specific
        'has_abbrev_name': convert_value(re.search(r'(\w+ )?\w\. \w+', row['text']) is not None),
        'has_typical_name_form': convert_value(re.search(r'(\w+ (\w\. )?\w+($|,))+', row['text']) is not None),
        'has_academic_title': convert_value(
            re.search(r'(^| |\W)(dr\.?|m\.?d\.?|ph\.?d\.?)(\W| |$)', row['text'], re.IGNORECASE) is not None),
        # date specific
        'contains_month': convert_value(
                any(month.lower() in row['text'].lower() for month in month_names)),
        'contains_short_month': convert_value(
                any(month.lower() in row['text'].lower() for month in month_short_names)),
        'has_daytime': convert_value(re.search(r'\d{2}:\d{2}', row['text']) is not None),
        'has_am_pm': convert_value(re.search(r'( |\d)(am|AM|pm|PM)($| |\W)', row['text']) is not None),
        'has_consec_digits=2': convert_value(re.search(r'\D\d{2}\D|^\d{2}\D|\D\d{2}$', row['text']) is not None),
        'has_consec_digits=4': convert_value(re.search(r'\D\d{4}\D|^\d{4}\D|\D\d{4}$', row['text']) is not None),
        'has_dateformat=a': convert_value(re.search(r'\d{2}.\d{2}.\d{4}', row['text']) is not None),
        'has_dateformat=b': convert_value(re.search(r'\d{4}.\d{2}.\d{2}', row['text']) is not None),
        'has_dateformat=c': convert_value(re.search(r'\d{2}.\w+?.\d{4}', row['text']) is not None),
        'has_dateformat=d': convert_value(re.search(r'\d{4}.\w+?.\d{2}', row['text']) is not None),
        'has_dateformat=e': convert_value(re.search(r'\w+.\d{1,2}..?\d{4}', row['text']) is not None),

        # text styling features
        'font_size': convert_value(row['font_size'].rstrip('px')),
        'font_size=' + font_size_bucket(row['font_size']): convert_value(1),
        'font_weight=bold': convert_value('bold' == parse_font_weight(row['font_weight'])),
        'font_weight=normal': convert_value('normal' == parse_font_weight(row['font_weight'])),
        'font_style=normal': convert_value(row['font_style'] == 'normal'),
        'font_style=italic': convert_value(row['font_style'] == 'italic'),
        'text_align=left': convert_value(text_alignment(row['text_align'], 'left')),
        'text_align=center': convert_value(text_alignment(row['text_align'], 'center')),
        'text_align=justify': convert_value(text_alignment(row['text_align'], 'justify')),
        'text_align=right': convert_value(text_alignment(row['text_align'], 'right')),

        # html element features
        'has_id': convert_value(len(row['id']) > 0),
        'id_group=' + parse_identifiers(row['id']): convert_value(1),
        'has_rules': convert_value(len(row['rules']) > 0),
        'h_tag': convert_value(row['node_type'].lower() in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']),

        # element style features
        'background_color': convert_value(parse_colour(row['background_color'])),
        'background_color_r': convert_value(parse_colour(row['background_color'], part=0)),
        'background_color_g': convert_value(parse_colour(row['background_color'], part=1)),
        'background_color_b': convert_value(parse_colour(row['background_color'], part=2)),
        'has_background_image': convert_value(row['background_image'] != 'none'),
        'has_background_color=white': convert_value(
            row['background_color'] == 'rgb(255, 255, 255)' or row['background_color'] == 'rgba(0, 0, 0, 0)'),
        'has_background_color=black': convert_value(row['background_color'] == 'rgb(0, 0, 0)'),

        # styling features
        'css_width': convert_value(int(float(row['css_width'].rstrip('px')))
                                   if row['css_width'] != 'auto' and '%' not in row['css_width']
                                   else None),
        'css_width_rel': convert_value(int(float(row['css_width'].rstrip('%')))
                                       if row['css_width'] != 'auto' and '%' in row['css_width']
                                       else None),
        'css_height': convert_value(int(float(row['css_height'].rstrip('px')))
                                    if row['css_height'] != 'auto' and '%' not in row['css_height']
                                    else None),
        'css_height_rel': convert_value(int(float(row['css_height'].rstrip('%')))
                                        if row['css_height'] != 'auto' and '%' in row['css_height']
                                        else None),

        # spacing features
        'padding_top': convert_value(row['padding_top'].rstrip('px').rstrip('%')),
        'padding_left': convert_value(row['padding_left'].rstrip('px').rstrip('%')),
        'padding_bottom': convert_value(row['padding_bottom'].rstrip('px').rstrip('%')),
        'padding_right': convert_value(row['padding_right'].rstrip('px').rstrip('%')),
        'margin_top': convert_value(row['margin_top'].rstrip('px').rstrip('%')),
        'margin_left': convert_value(row['margin_left'].rstrip('px').rstrip('%')),
        'margin_bottom': convert_value(row['margin_bottom'].rstrip('px').rstrip('%')),
        'margin_right': convert_value(row['margin_right'].rstrip('px').rstrip('%'))
    }

    # lower right corner point
    features['h_pos_right'] = convert_value(features['h_pos'] + features['width'])
    features['v_pos_bottom'] = convert_value(features['v_pos'] + features['height'])

    # center x/y
    features['center_x_raw'] = convert_value(features['h_pos'] + (features['width'] / 2))
    features['center_y_raw'] = convert_value(features['v_pos'] + (features['height'] / 2))

    # height/width ratio and implications
    features['hw_ratio'] = convert_value(features['height'] / (features['width'] + 0.0001))
    features['hw_ratio=portrait'] = convert_value(features['hw_ratio'] >= 1.1)
    features['hw_ratio=port_slim'] = convert_value(features['hw_ratio'] > 2.7)
    features['hw_ratio=landscape'] = convert_value(0.35 < features['hw_ratio'] <= 0.9)
    features['hw_ratio=land_slim'] = convert_value(0.25 < features['hw_ratio'] <= 0.35)
    features['hw_ratio=land_slim<0.25'] = convert_value(0.15 < features['hw_ratio'] <= 0.25)
    features['hw_ratio=land_slim<0.15'] = convert_value(0.08 < features['hw_ratio'] <= 0.15)
    features['hw_ratio=land_slim<0.08'] = convert_value(features['hw_ratio'] <= 0.08)
    features['hw_ratio=square'] = convert_value(0.9 < features['hw_ratio'] < 1.1)

    features['area'] = convert_value(features['height'] * features['width'])
    features['screen_area'] = convert_value(100.0 * (features['area'] / (screen_size[1] * screen_size[0])))

    # conditional text syntax features
    features['has_dots'] = convert_value(features['num_dots'] > 0)
    features['has_digits'] = convert_value(features['num_digits'] > 0)
    features['has_uppercase'] = convert_value(features['num_uppercase_chars'] > 0)
    features['has_linebreaks'] = convert_value(features['num_linebreaks'] > 0)

    # number of words normalisation
    features['num_words=1'] = convert_value(features['num_words'] == 1)
    features['num_words=2'] = convert_value(features['num_words'] == 2)
    features['num_words=3'] = convert_value(features['num_words'] == 3)
    features['num_words=3-5'] = convert_value(3 < features['num_words'] <= 5)
    features['num_words=5-8'] = convert_value(5 < features['num_words'] <= 8)
    features['num_words=8-12'] = convert_value(8 < features['num_words'] <= 12)
    features['num_words>12'] = convert_value(features['num_words'] > 12)

    # same for uppercase words
    features['num_uppercase_words=1'] = convert_value(features['num_uppercase_words'] == 1)
    features['num_uppercase_words=2'] = convert_value(features['num_uppercase_words'] == 2)
    features['num_uppercase_words=3'] = convert_value(features['num_uppercase_words'] == 3)
    features['num_uppercase_words=3-5'] = convert_value(3 < features['num_uppercase_words'] <= 5)
    features['num_uppercase_words=5-8'] = convert_value(5 < features['num_uppercase_words'] <= 8)
    features['num_uppercase_words=8-12'] = convert_value(8 < features['num_uppercase_words'] <= 12)
    features['num_uppercase_words>12'] = convert_value(features['num_uppercase_words'] > 12)

    # combined spacing features
    features['space_top'] = convert_value(features['padding_top'] + features['margin_top'])
    features['space_left'] = convert_value(features['padding_left'] + features['margin_left'])
    features['space_bottom'] = convert_value(features['padding_bottom'] + features['margin_bottom'])
    features['space_right'] = convert_value(features['padding_right'] + features['margin_right'])

    # Element in top ... (horizontal slice)
    features['in_top_20'] = convert_value(0 <= features['v_pos_rel'] < 0.2)
    features['in_top_40'] = convert_value(0.2 <= features['v_pos_rel'] < 0.4)
    features['in_top_60'] = convert_value(0.4 <= features['v_pos_rel'] < 0.6)
    features['not_top_40'] = convert_value(features['v_pos_rel'] >= 0.4)
    features['not_top_60'] = convert_value(features['v_pos_rel'] >= 0.6)

    # In which quadrant of the first screen is the element
    quadrant_raster = [2, 2]
    quadrant_raster_blocks = [1.0 / quadrant_raster[0], 1.0 / quadrant_raster[1]]
    for qr_i in range(quadrant_raster[0]):
        for qr_j in range(quadrant_raster[1]):
            features['in_quadrant=' + str(qr_i) + str(qr_j)] = convert_value(
                    ((qr_i * quadrant_raster_blocks[0]) <= features['v_pos_rel'] < (
                    (qr_i + 1) * quadrant_raster_blocks[0])) and
                    ((qr_j * quadrant_raster_blocks[1]) <= features['h_pos_rel'] < (
                    (qr_j + 1) * quadrant_raster_blocks[1])))

    # In which screen is the element 
    features['in_screen=1'] = convert_value((0 <= features['v_pos_rel'] < 1) and (0 <= features['h_pos_rel'] < 1))
    features['in_screen=2'] = convert_value((1 <= features['v_pos_rel'] < 2) and (0 <= features['h_pos_rel'] < 1))
    features['in_screen=3'] = convert_value((2 <= features['v_pos_rel'] < 3) and (0 <= features['h_pos_rel'] < 1))
    features['in_screen=4'] = convert_value((3 <= features['v_pos_rel'] < 4) and (0 <= features['h_pos_rel'] < 1))
    features['in_screen=5'] = convert_value((4 <= features['v_pos_rel'] < 5) and (0 <= features['h_pos_rel'] < 1))
    features['in_screen=6'] = convert_value((5 <= features['v_pos_rel'] < 6) and (0 <= features['h_pos_rel'] < 1))
    features['in_screen>6'] = convert_value((features['v_pos_rel'] >= 6) and (0 <= features['h_pos_rel'] < 1))

    # in screen vertical relative position
    features['v_pos_srel'] = convert_value(features['v_pos_rel'] - int(features['v_pos_rel']))

    # Element width is ... % of page
    features['width_rel<10'] = convert_value(features['width_rel'] < 0.1)
    features['width_rel<20'] = convert_value(0.1 <= features['width_rel'] < 0.2)
    features['width_rel<30'] = convert_value(0.2 <= features['width_rel'] < 0.3)
    features['width_rel<40'] = convert_value(0.3 <= features['width_rel'] < 0.4)
    features['width_rel>40'] = convert_value(features['width_rel'] >= 0.4)

    # extract font-family features
    # todo add merge of both
    families, fam_groups = parse_font_family(row['font_family'])
    if len(fam_groups['family']) > 0:
        features['font_family=' + fam_groups['family'][0]] = convert_value(1)
    if len(fam_groups['weight']) > 0:
        features['font_weight_fam=' + parse_font_weight(fam_groups['weight'][0])] = convert_value(1)
    if len(fam_groups['style']) > 0:
        features['font_style_fam=' + fam_groups['style'][0]] = convert_value(1)

    # node type
    allowed_types = ['a', 'span', 'p', 'td', 'li', 'strong', 'h2', 'h3', 'b',
                     'font', 'em', 'h4', 'h1', 'i', 'label', 'time', 'small',
                     'th', 'button', 'cite', 'h5', 'dt', 'big', 'nobr',
                     'figcaption', 'h6', 'dd', 'data', 'u']
    for tp in allowed_types:
        features['node_type=' + tp] = convert_value(row['node_type'].lower() == tp)
    # fallback type: div
    features['node_type=div'] = convert_value(row['node_type'].lower() not in allowed_types)

    return features


def rescale(row, agg, field):
    return (((2.0 * row[field]) - agg[field]['max'] - agg[field]['min']) / (agg[field]['max'] - agg[field]['min']))


def standardise(row, agg, field):
    return ((row[field] - agg[field]['mean']) / agg[field]['std'])


def add_relative_features(row, agg, frm, ofrm, nn):
    # print(row)
    orow = ofrm.loc[row.name]

    max_neighbours = 6
    nneighbours = max_neighbours if len(frm) > max_neighbours else len(frm)
    dists, neighs = nn.kneighbors(row[['center_x_raw', 'center_y_raw']].as_matrix().reshape(1, -1), nneighbours)
    dists = dists[0][1:]  # reshape and remove self
    neighs = frm.iloc[neighs[0][1:]]  # reshape, remove self and link back with frame

    page_width = agg['h_pos_right']['max'] - agg['h_pos']['min']
    page_height = agg['v_pos_bottom']['max'] - agg['v_pos']['min']
    page_area = page_width * page_height

    # rescaled continuous features
    row['v_pos_rescaled'] = convert_value(rescale(row, agg, 'v_pos'))
    row['h_pos_rescaled'] = convert_value(rescale(row, agg, 'h_pos'))
    row['width_rescaled'] = convert_value(rescale(row, agg, 'width'))
    row['height_rescaled'] = convert_value(rescale(row, agg, 'height'))
    row['font_size_rescaled'] = convert_value(rescale(row, agg, 'font_size'))
    row['area_rescaled'] = convert_value(rescale(row, agg, 'area'))
    row['num_words_rescaled'] = convert_value(rescale(row, agg, 'num_words'))
    row['text_length_rescaled'] = convert_value(rescale(row, agg, 'text_length'))

    # standardised continuous features
    row['v_pos_standardised'] = convert_value(standardise(row, agg, 'v_pos'))
    row['h_pos_standardised'] = convert_value(standardise(row, agg, 'h_pos'))
    row['width_standardised'] = convert_value(standardise(row, agg, 'width'))
    row['height_standardised'] = convert_value(standardise(row, agg, 'height'))
    row['font_size_standardised'] = convert_value(standardise(row, agg, 'font_size'))
    row['area_standardised'] = convert_value(standardise(row, agg, 'area'))
    row['num_words_standardised'] = convert_value(standardise(row, agg, 'num_words'))

    # position relative to page dimension (not relative to screen dimension)
    row['v_pos_prel'] = convert_value((row['v_pos'] - agg['v_pos']['min']) / agg['v_pos']['max'])
    row['h_pos_prel'] = convert_value((row['h_pos'] - agg['h_pos']['min']) / agg['h_pos']['max'])

    # horizontal difference from center line of page
    row['h_pos_from_mean'] = convert_value(row['h_pos'] - agg['h_pos']['mean'])
    row['h_pos_rel_from_mean'] = convert_value(row['h_pos_rel'] - agg['h_pos_rel']['mean'])

    # width in ten slices, rel to mean
    row['h_pos_from_mean=-50--35'] = convert_value(row['h_pos_from_mean'] / page_width < -0.35)
    row['h_pos_from_mean=-35--20'] = convert_value(-0.35 <= row['h_pos_from_mean'] / page_width < -0.2)
    row['h_pos_from_mean=-20--5'] = convert_value(-0.2 <= row['h_pos_from_mean'] / page_width < -0.05)
    row['h_pos_from_mean=-5-5'] = convert_value(-0.05 <= row['h_pos_from_mean'] / page_width < 0.05)
    row['h_pos_from_mean=5-20'] = convert_value(0.05 <= row['h_pos_from_mean'] / page_width < 0.2)
    row['h_pos_from_mean=20-35'] = convert_value(0.2 <= row['h_pos_from_mean'] / page_width < 0.35)
    row['h_pos_from_mean=35-50'] = convert_value(0.35 <= row['h_pos_from_mean'] / page_width)

    # position clipped to the left side
    row['h_pos_clipped'] = convert_value(row['h_pos'] - agg['h_pos']['min'])
    row['v_pos_clipped'] = convert_value(row['v_pos'] - agg['v_pos']['min'])
    row['h_pos_clipped_right'] = convert_value(row['h_pos_right'] - agg['h_pos']['min'])
    row['v_pos_clipped_bottom'] = convert_value(row['v_pos_bottom'] - agg['v_pos']['min'])

    # center point of element (page clipped to topleft)
    row['center_x'] = convert_value(row['h_pos_clipped'] + (row['width'] / 2))
    row['center_y'] = convert_value(row['v_pos_clipped'] + (row['height'] / 2))

    row['page_area'] = convert_value(
        (agg['h_pos']['max'] - agg['h_pos']['min']) * (agg['v_pos']['max'] - agg['v_pos']['min']))
    row['page_v_screenspan'] = convert_value(agg['v_pos']['max'] / screen_size[1])

    # page relative width
    row['width_prel'] = convert_value(row['width'] / page_width)
    # Element width is ... % of page
    row['width_prel<10'] = convert_value(row['width_prel'] < 0.1)
    row['width_prel<20'] = convert_value(0.1 <= row['width_prel'] < 0.2)
    row['width_prel<30'] = convert_value(0.2 <= row['width_prel'] < 0.3)
    row['width_prel<40'] = convert_value(0.3 <= row['width_prel'] < 0.4)
    row['width_prel<50'] = convert_value(0.4 <= row['width_prel'] < 0.5)
    row['width_prel>50'] = convert_value(row['width_prel'] >= 0.5)

    # font size diff to average
    row['font_size_diff'] = convert_value(row['font_size'] - agg['font_size']['mean'])
    row['font_size>mean'] = convert_value(1 < row['font_size_diff'] <= 5)
    row['font_size>>mean'] = convert_value(row['font_size_diff'] > 5)
    row['font_size<mean'] = convert_value(-3 < row['font_size_diff'] < -1)
    row['font_size<<mean'] = convert_value(row['font_size_diff'] < -2.5)
    row['font_size==mean'] = convert_value(row['font_size_diff'] == 0)
    row['font_size=mean'] = convert_value(-1 <= row['font_size_diff'] <= 1)

    font_rank = frm['font_size'].nlargest(n=3)
    row['is_largest_font'] = convert_value(font_rank.iloc[0] == row['font_size'])
    row['is_second_largest_font'] = convert_value(len(font_rank) > 1 and font_rank.iloc[1] == row['font_size'])
    row['is_smallest_font'] = convert_value(agg['font_size']['min'] == row['font_size'])

    stats = neighs[['font_size']].describe()
    row['nb_font_size_diff'] = convert_value(row['font_size'] - stats['font_size']['mean'])
    row['nb_font_size>mean'] = convert_value(row['nb_font_size_diff'] > 0)
    row['nb_font_size>>mean'] = convert_value(row['nb_font_size_diff'] > 5)
    row['nb_font_size<mean'] = convert_value(row['nb_font_size_diff'] < 0)
    row['nb_font_size<<mean'] = convert_value(row['nb_font_size_diff'] < 2)
    row['nb_font_size==mean'] = convert_value(row['nb_font_size_diff'] == 0)
    row['nb_font_size=mean'] = convert_value(-1 <= row['nb_font_size_diff'] <= 1)

    avg_dist_perc = [
        [],  # skip
        [],  # skip
        [45.0, 80.0, 110.0, 145.0, 180.0, 220.0],  # k=2
        [],  # skip
        [60.0, 100.0, 170.0, 210.0, 250.0, 300.0]  # k=4
    ]
    for k in range(1, nneighbours):
        avg_dist = sum(dists[:k]) / k
        row['avg_neighbourdist_k=' + str(k)] = convert_value(avg_dist)
        if k == 2 or k == 4:
            for c, val in enumerate(avg_dist_perc[k]):
                if c == 0:
                    row['avg_neighbourdist_k=' + str(k) + '<' + str(val)] = convert_value(
                        avg_dist < avg_dist_perc[k][c])
                else:
                    row['avg_neighbourdist_k=' + str(k) + '<' + str(val)] = convert_value(
                        avg_dist_perc[k][c - 1] <= avg_dist < avg_dist_perc[k][c])
                if c == len(avg_dist_perc[k]) - 1:
                    row['avg_neighbourdist_k=' + str(k) + '>' + str(val)] = convert_value(
                        avg_dist > avg_dist_perc[k][c])
    return row


def make_mask(feature_frame):
    full_mask = [True] * len(feature_frame)
    full_mask &= (feature_frame['v_pos'] != 0) | (feature_frame['h_pos'] != 0)
    full_mask &= (feature_frame['v_pos'] >= 0) & (feature_frame['h_pos'] >= 0)
    full_mask &= (feature_frame['h_pos_rel'] < 1)
    full_mask &= (feature_frame['v_pos_rel'] < 6)
    full_mask &= (feature_frame['width'] > 0)
    full_mask &= (feature_frame['height'] > 0)
    full_mask &= (feature_frame['height'] < 200)
    full_mask &= (feature_frame['width'] < 1700)
    full_mask &= (feature_frame['text_length'] >= 3)
    return full_mask
    
    
def feature_function(raw_frame):
    # get basic features
    frame = pd.DataFrame([get_features(row) for row in raw_frame.to_dict(orient='records')],
                         index=raw_frame.index.tolist())

    mask = make_mask(frame)

    # build nearest neighbour map of elements
    nn = NearestNeighbors().fit(frame[mask][['center_x_raw', 'center_y_raw']].as_matrix())

    # enrich features
    return mask, frame[mask].apply(add_relative_features, axis=1, agg=frame.describe(), frm=frame, ofrm=raw_frame, nn=nn)
