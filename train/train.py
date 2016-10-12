from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import warnings
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.regularizers import l2
import keras.optimizers as opts
from .sms_pandas import DataHolder
from .sms_pandas import label_dict
from .sms_eval import Evaluator
from ..libs.selected_features import feature_weights_b

# data prep params
unassigned_reduction = 0.20
neighbourhood_size = 10

# Neural network training params
nb_epoch = 25
batch_size = 10
val_data_size = 50

# Data init
mask_params = {'exclude_origin': True, 'exclude_neg': True, 'exclude_outside_x': True, 'exclude_outside_y': True,
               'exclude_zero_width': True, 'exclude_zero_height': True, 'exclude_large_height': True,
               'exclude_large_width': True, 'exclude_onechar': True, 'exclude_twochar': True,
               'exclude_threechar': True, 'has_title': True, 'has_author': False, 'has_issued': False,
               'spread': True}

holder = DataHolder('./learnbase', fix_labeling=False, from_fix=False, create_features=True,
                    tmp_folder='./tmp', features_from_tmp=True, add_neighbourhoods=True,
                    reload_tmp=False, mask_params=mask_params, threads=20)

raw_frame = holder.get_raw_frame()
meta_frame = holder.get_meta_frame()
feature_frame = holder.get_feature_frame()

mask = holder.get_mask(**mask_params)

# list the dictionaries for reference
print("Features:")
print(feature_frame.columns)
print("Labels:")
print(label_dict)

print("All initialised!")

eva = Evaluator(feature_frame, mask, feature_weights_b)
keep_features = eva.get_keep_features()
test, train, X_train, y_train, Y_train, X_test, y_test, Y_test, nb_classes, class_weights = eva.get_traintest_i_flat(
        reduction_min=4,
        reduction=unassigned_reduction)

# train neural network
model_nn1 = Sequential()
model_nn1.add(Dense(output_dim=len(keep_features), input_dim=len(keep_features), init='he_uniform'))
model_nn1.add(Activation("linear"))

model_nn1.add(Dense(80, W_regularizer=l2(0.01), init='he_uniform'))
model_nn1.add(Activation("softsign"))
model_nn1.add(Dropout(0.4))

model_nn1.add(Dense(35, W_regularizer=l2(0.01), init='he_uniform'))
model_nn1.add(Activation("softsign"))
model_nn1.add(Dropout(0.2))

model_nn1.add(Dense(nb_classes, init='he_uniform'))
model_nn1.add(Activation("softmax"))

model_nn1.summary()

model_nn1.compile(loss='msle',
                  optimizer=opts.Adadelta(),
                  metrics=['accuracy'])

history_nn1 = model_nn1.fit(X_train, Y_train,
                            batch_size=batch_size,
                            nb_epoch=nb_epoch,
                            class_weight=class_weights,
                            verbose=1,
                            validation_data=(X_test[:val_data_size], Y_test[:val_data_size]))
print('done')

# save neural network
model_nn1.save('../models/nn1_model_25epochs_10batch_sorted.h5')

# show some small eval
predictions_nn1 = model_nn1.predict_proba(X_test)
predictframe_nn1 = eva.get_predictframe(predictions_nn1, global_idxmax=True, plabel_idxmax=True)

print('raw')
eva.measure_performance(None, y_test, predictions_nn1, show_curves=False)
print('idxmax')
eva.measure_performance(None, y_test, list(predictframe_nn1['gilabel']))
print('plabel idxmax')
eva.measure_performance(None, y_test, list(predictframe_nn1['ilabel']))

test, train, X_train, y_train, Y_train, X_test, y_test, Y_test, nb_classes, class_weights = eva.get_traintest_i_flat(
        reduction_min=4,
        reduction=0.6,
        testsetfix=list(set(test['instance'])))

# build random forest
forest = RandomForestClassifier(class_weight='balanced', criterion='entropy', n_estimators=50, n_jobs=10)
bag = BaggingClassifier(forest, n_estimators=10, max_features=0.7, n_jobs=1)
clf = OneVsRestClassifier(bag, n_jobs=1)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    clf.fit(X_train, y_train)
    predictions_rf = clf.predict_proba(X_test)

# save the random forest
joblib.dump(clf, '../models/rf_model_50trees_10bag_4ovsr_v018_sorted.pkl', compress=True)

# do some small eval
predictframe_rf = eva.get_predictframe(predictions_rf, global_idxmax=True, plabel_idxmax=True)

print('raw')
eva.measure_performance(None, y_test, predictions_rf, show_curves=True)
print('idxmax')
eva.measure_performance(None, y_test, list(predictframe_rf['gilabel']))
print('plabel idxmax')
eva.measure_performance(None, y_test, list(predictframe_rf['ilabel']))
