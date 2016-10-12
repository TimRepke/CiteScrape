# CiteScrape

This is an experimental demonstration of results found in my master's thesis about the "Extraction of Citation Data from
WebPages Based on Visual Cues". It provides a simple web frontend, where you can paste a link, which is rendered in
an iframe and will (hopefully) retrieve the title, author and publishing date of whatever the link pointed to.

These meta-data are needed to cite that online resource and can be used by a citation processor to generate a properly
formatted bibliographic entry. This repository contains trained models, which are trained on roughly 600 webpages.
Testing results on about 200 more pages show following performance (10-fold cross-evaluated):

||Precision|Recall|F-Score|Support|
|---|---|---|---|---|
|Title  | 0.93 | 0.82 | 0.87 | 189|
|Author | 0.61 | 0.50 | 0.55 | 118|
|Date   | 0.90 | 0.66 | 0.76 | 97|

## Setup
1. Clone the repository
2. `cd` into the project root directory
3. create and load virtual env (optional)
```
virtualenv -p /usr/bin/python3.5 csenv
source csenv/bin/activate
```
For help check http://docs.python-guide.org/en/latest/dev/virtualenvs/
4. install requirements
```
pip install --upgrade requirements.txt
```
5. run the server
```
KERAS_BACKEND=theano FLASK_DEBUG=1 FLASK_APP=main.py flask run
```
Note: The neural network was trained with the TensorFlow backend using Keras, which should be fully compatible (in this
setup) with Theano.
6. Open this URL in your browser: http://localhost:5000/