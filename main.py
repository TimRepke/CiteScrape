from flask import Flask
from flask import request
from flask import render_template
from flask import jsonify
import re
import traceback
import urllib.request
import urllib.error

app = Flask(__name__)

# url_for('static', filename='style.css')


@app.route('/')
def hello_world():
    return render_template('extract.html')


@app.route('/extract/', methods=['POST'])
def extract():
    app.logger.info('extraction request received')
    page = request.get_json()
    import time
    time.sleep(5)

    return jsonify(**page)


@app.route('/mirror/', methods=['GET', 'POST'])
def mirror():
    url = None

    try:
        if request.method == 'POST':
            url = request.form['url']
        elif request.method == 'GET':
            url = request.args.get('url', '')

        if not url:
            return "Please enter URL."

        app.logger.info('Request to proxy URL: %s' % url)

        req = urllib.request.Request(url)
        response = urllib.request.urlopen(req)
        page = response.read().decode("utf-8")

        regex = re.compile('<head(.*?)>', re.IGNORECASE)
        page = regex.sub(r'<head\1><base href="%s" />' % url, page, count=1)

        return page

    except urllib.error.HTTPError as e:
        app.logger.warn("HTTPError occurred: %s" % e.code)
        app.logger.warn(e.read())
        return "Nope... %s" % e.code
    except Exception as e:
        app.logger.warn('Some error occurred.')
        app.logger.warn(traceback.format_exc())
        return "<pre>" + traceback.format_exc() + "</pre>" if app.debug else "Some error occurred, sorry."


#     app.get('/', function(req, res) {
#   if (req.query.url) {
#     var url =  req.query.url;
#     logger.info('Proxy request for: ' + url);
#
#     utils.requestURL({
#       url: url
#     }).then(function(body) {
#       logger.trace('Response back in mirror route');
#       var $ = cheerio.load(body);
#       $('head').prepend(
#           '<base href="' + url + '" />' +
#           '<style type="text/css">' +
#           '   .test-hovered{' +
#           '       border: 3px dashed red !important;' +
#           '       cursor: copy;' +
#           '   }' +
#           '</style>');
#
#       res.set({ 'content-type': 'text/html;charset=utf-8' });
#       res.send($.html());
#     }, function(error) {
#       logger.trace('Response back in mirror route as an error');
#       res.status(400).send({ error: 'Error appeared: ' + error });
#     });
#   } else {
#     res.status(400).send({ error: 'No URL specified!' });
#   }
# });