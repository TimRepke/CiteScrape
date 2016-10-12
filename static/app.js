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
    constructor(target) {
        // some singleton action
        if(LoadingSpinner.targetNames && LoadingSpinner.targetNames.indexOf(target) >= 0){
            return LoadingSpinner.instances[LoadingSpinner.targetNames.indexOf(target)];
        } else {
            LoadingSpinner.targetNames = (LoadingSpinner.targetNames || [])
            LoadingSpinner.targetNames.push(target);
            LoadingSpinner.instances = (LoadingSpinner.instances || [])
            LoadingSpinner.instances.push(this);
        }
        this.instance = this;
        this.cnt = 0;
        this.loop = null;
        this.target = document.getElementById(target);
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
    var loadingSpinner = new LoadingSpinner('refme_result');
    loadingSpinner.start();

    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (this.readyState == 4) {
            loadingSpinner.stop();
            if (this.status == 200) {
                var result = JSON.parse(this.responseText);
                log(result);
                document.getElementById('refme_result').innerHTML = "<h3>RefME</h3><pre>" + JSON.stringify(result, null, 2) + "</pre>";
            } else {
                document.getElementById('refme_result').innerText = "Some error occurred :-(";
            }
        }
    };
    xhr.open('GET', 'https://refmenode-production.herokuapp.com/search?type=website&query='+encodeURIComponent(url), true);
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

// rgb(191, 226, 182)
var TARGET_GREEN = [191, 226, 182];
// rgb(228, 129, 129)
var TARGET_RED = [228, 129, 129];

function getConfidenceColour(confidence) {
    confidence = confidence*1.7;
    return 'rgb('+(TARGET_RED[0]+((TARGET_GREEN[0]-TARGET_RED[0])*confidence)).toFixed(0)+','+
                  (TARGET_RED[1]+((TARGET_GREEN[1]-TARGET_RED[1])*confidence)).toFixed(0)+','+
                  (TARGET_RED[2]+((TARGET_GREEN[2]-TARGET_RED[2])*confidence)).toFixed(0)+')';
}

function getCell(score) {
    return '<td style="background-color:'+getConfidenceColour(score)+';">'+shortenNumber(score)+'</td>'
}

function getFullTable(scores) {
    // conf [t,a,d,u], l [fin,gilab, ilab, plab], text
    var ret = '<table border="1"> <tr><th>Text</th><th>fin</th><th>gilab</th><th>ilab</th><th>plab</th><th>title</th><th>author</th><th>date</th><th>none</th></tr>';
    for (var score of scores) {
        ret += '<tr><td>'+shortenStr(score['text'], 40)+'</td><td>'+score['label']['final']+'</td><td>'+score['label']['gilabel']+
               '</td><td>'+score['label']['ilabel']+'</td><td>'+score['label']['plabel']+'</td>'+
               getCell(score['confidence']['title'])+getCell(score['confidence']['author'])+
               getCell(score['confidence']['date'])+getCell(score['confidence']['unassigned'])+'</tr>'
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
    // start loading "animation"
    var loadingSpinner = new LoadingSpinner('result');
    loadingSpinner.start();

    // remove table (if existing)
    document.getElementById('result_table').innerHTML = ''

    // fetch all elements from the page in the iframe
    var elements = fetchElements(document.getElementById('target_url').value,
                                  document.getElementById('scrapesite').contentWindow,
                                  document.getElementById('scrapesite').contentWindow.document);
    log(elements);


    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (this.readyState == 4) {
            loadingSpinner.stop();
            if (this.status == 200) {
                // parse the result
                var result = JSON.parse(this.responseText);
                resultStore = result;
                log(result);

                // dump result into the left column
                document.getElementById('result').innerHTML =
                    '<h3>CiteScrape</h3><pre>' + JSON.stringify(result['clean'], null, 2) + '</pre>'+
                    '<button onclick="showFullTable(\'merged\')">Merged</button>'+
                    '<button onclick="showFullTable(\'neural_net\')">Neural Net</button>'+
                    '<button onclick="showFullTable(\'random_forest\')">Random Forest</button>';

                // show the detailed table for all candidates
                showFullTable('merged')
            } else {
                document.getElementById('result').innerText = 'Some error occurred :-(';
            }
        }
    };

    // send list of elements as JSON to the server
    xhr.open('POST', defaultBaseUrl + defaultScrapeRoute, true);
    xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
    xhr.send(JSON.stringify(elements));

    // start RefME scraper
    refmeResult(document.getElementById('target_url').value)
}