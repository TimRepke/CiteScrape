const defaultBaseUrl = 'http://localhost:5000';
const defaultMirrorRoute = '/mirror/?url=';
const defaultScrapeRoute = '/extract/';

const skipElements = ['BR', 'HR', 'SCRIPT', 'NOSCRIPT', 'STYLE'];
const styleAttributes = ['background-color', 'background-image', 'height', 'width',
            'padding-top', 'padding-bottom', 'padding-left', 'padding-right',
            'margin-top', 'margin-bottom', 'margin-left', 'margin-right',
            'font-family', 'font-size', 'font-weight', 'font-style', 'text-align'];

log = console.log

class LoadingSpinner {
    constructor() {
        // some singleton action
        if(LoadingSpinner.inst){
            return LoadingSpinner.inst;
        }
        this.instance = this;
        this.cnt = 0;
        this.loop = null;
        this.target = document.getElementById('result');
        LoadingSpinner.inst = this;
    }

    start() {
        log(this.loop)
        if (this.loop === null){
            this.cnt = 0;
            var that = this;
            this.loop = setInterval(function() {
                that.target.innerText = 'Loading' + '.'.repeat(that.cnt % 4 + 1);
                that.cnt++;
            }, 400)
        }
    }

    stop() {
        clearInterval(this.loop);
        this.loop = null;
    }
}

function fetchElements(url, iFrameWindow, iFrameDoc){
    var output = {
        url: url,
        retrieved_at: new Date,
        elements: []
    };

    var el, innerText, ruleList, ref2, l, i, style, bounds, tmp;

    var elements = iFrameDoc.getElementsByTagName('body')[0].getElementsByTagName("*");

    for (var j = 0, len = elements.length; j < len; j++) {
        // some handlers
        el = elements[j];
        innerText = getInnerText(el);

        // skip empty or SKIP elements
        if (skipElements.indexOf(el.nodeName) >= 0 || !innerText) continue;

        // get element boundary
        bounds = el.getBoundingClientRect();

        // prepare information for this element
        tmp = {
            id: el.id,
            className: el.className,
            nodeName: el.nodeName,
            text: innerText,
            html: el.innerHTML,
            bounds: {
                height: el.offsetHeight || 0,
                width: el.offsetWidth || 0,
                top: bounds.top || 0,
                left: bounds.left || 0
            },
            style: {},
            rules: []
        };

        // get computed style and add relevant info to tmp.style
        style = iFrameWindow.getComputedStyle(el);
        for (i = 0; i < styleAttributes.length; i++) {
            tmp.style[styleAttributes[i]] = style.getPropertyValue(styleAttributes[i]);
        }

        // check, if element is hidden (if so, skip)
        if (style.getPropertyValue('visibility') === 'hidden' ||
            style.getPropertyValue('display') === 'none') continue;

        // get applied CSS rules and add to tmp.rules
        ruleList = el.ownerDocument.defaultView.getMatchedCSSRules(el, '') || [];
        for (i = l = 0, ref2 = ruleList.length; 0 <= ref2 ? l < ref2 : l > ref2; i = 0 <= ref2 ? ++l : --l) {
            tmp.rules.push(ruleList[i].selectorText);
        }

        // push tmp into the output
        output.elements.push(tmp);
    }

    // return result
    return output;

    /**
     * Returns the inner text without content of child elements of an element
     * @param e the element
     * @returns {string} inner text
     */
    function getInnerText(e) {
        var run = e.firstChild;
        var texts = [];
        while (run) {
            if (run.nodeType === 3) {
                texts.push(run.data);
            }
            run = run.nextSibling;
        }
        var text = texts.join('').replace(/\s+/g, ' ');
        var tagRegex = /(<[^>]*?\/>|<[^>]*>[^<]*<\/[^>]*>)/gmi;

        while (tagRegex.test(text)) {
            text = text.replace(tagRegex, '');
        }

        return text.trim();
    }
}

function loadIFrame() {
    document.getElementById('scrapesite').src = defaultBaseUrl + defaultMirrorRoute +
                                                encodeURIComponent(document.getElementById('target_url').value);
}

function refmeResult(url) {
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (this.readyState == 4) {
            if (this.status == 200) {
                var result = JSON.parse(this.responseText);
                log(result);
                document.getElementById('refme_result').innerHTML = "<pre>" + JSON.stringify(result, null, 2) + "</pre>";
            } else {
                document.getElementById('refme_result').innerText = "Some error occurred :-(";
            }
        }
    };
    xhr.open('GET', 'http://scraper-service-staging.herokuapp.com/result?url='+encodeURIComponent(url), true);
    xhr.send();
}

function shortenStr(istr, len) {
    if (istr.length <= len) {
        return istr;
    } else {
        return istr.substr(0, len) + '&hellip;';
    }
}

function shortenNumber(num) {
    try {
        return num.toFixed(3);
    } catch(e) {
        return NaN;
    }
}

function getFullTable(scores) {
    // conf [t,a,d,u], l [fin,gilab, ilab, plab], text
    var ret = '<table border="1"> <tr><th>Text</th><th>fin</th><th>gilab</th><th>ilab</th><th>plab</th><th>title</th><th>author</th><th>date</th><th>none</th></tr>';
    for (var score of scores) {
        ret += '<tr><td>'+shortenStr(score['text'], 40)+'</td><td>'+score['label']['final']+'</td><td>'+score['label']['gilabel']+
               '</td><td>'+score['label']['ilabel']+'</td><td>'+score['label']['plabel']+'</td><td>'+shortenNumber(score['confidence']['title'])+
               '</td><td>'+shortenNumber(score['confidence']['author'])+'</td><td>'+shortenNumber(score['confidence']['date'])+
               '</td><td>'+shortenNumber(score['confidence']['unassigned'])+'</td></tr>'
    }
    ret += '</table>';
    return ret;
}

var resultStore = null;

function showFullTable(classifier) {
    if (!resultStore || !resultStore['all'][classifier]) {
        document.getElementById('result_table').innerHTML = 'Sorry, no data available';
    } else {
        document.getElementById('result_table').innerHTML = getFullTable(resultStore['all'][classifier])
    }
}

function scrape() {
    var loadingSpinner = new LoadingSpinner();
    loadingSpinner.start();

    refmeResult(document.getElementById('target_url').value)
    var elements = fetchElements(document.getElementById('target_url').value,
                                  document.getElementById('scrapesite').contentWindow,
                                  document.getElementById('scrapesite').contentWindow.document);
    log(elements);

    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (this.readyState == 4) {
            loadingSpinner.stop();
            if (this.status == 200) {
                var result = JSON.parse(this.responseText);
                resultStore = result;
                log(result);
                document.getElementById('result').innerHTML =
                    '<pre>' + JSON.stringify(result['clean'], null, 2) + '</pre>'+
                    '<button onclick="showFullTable(\'merged\')">Merged</button>'+
                    '<button onclick="showFullTable(\'neural_net\')">Neural Net</button>'+
                    '<button onclick="showFullTable(\'random_forest\')">Random Forest</button>';
                showFullTable('merged')
            } else {
                document.getElementById('result').innerText = 'Some error occurred :-(';
            }
        }
    };
    xhr.open('POST', defaultBaseUrl + defaultScrapeRoute, true);
    xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
    xhr.send(JSON.stringify(elements));
}