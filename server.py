import os
from flask import Flask, jsonify, send_from_directory, request, escape, render_template
import dataportraits
from pprint import pprint
import dataportraits.timers as timers
import time
import json
import random
import tokenizations
import datetime

REDIS_HOST = os.environ.get('DS_HOST', 'localhost')
REDIS_PORT = os.environ.get('DS_PORT', 8899)
SKETCH_NAME = os.environ.get('DS_NAME')
SKETCH_WIDTH = int(os.environ.get('DS_WIDTH'))

app = Flask(__name__, static_folder='./site/static', static_url_path='/static',
        template_folder='./site/templates')
sketch = dataportraits.RedisBFSketch(REDIS_HOST, REDIS_PORT, SKETCH_NAME, SKETCH_WIDTH)

MAX_LOG = 10000
@app.route("/q", methods=["POST", "GET"])
def query():
    results = []
    if 'document' in request.json:
        docs = [request.json['document']]
        # logging
        log_stamp = datetime.datetime.now().strftime("log-%Y-%m-%d")
        num_logged = sketch.redis_client.lpush(log_stamp, docs[0])
        if num_logged > MAX_LOG:
            sketch.redis_client.ltrim(log_stamp, -MAX_LOG, -1)

        # end logging
        with timers.Timer("Get Report") as t:
            report = sketch.contains_from_text(docs, sort_chains_by_length=True)[0]
        print(report['doc'])
        report['width'] = SKETCH_WIDTH
        report['seq_id'] = request.json['seq_id']
        report['segments'] = None
        return jsonify(report)
    else:
        return jsonify({})

@app.route("/overlap", methods=["POST"])
def overlap():
    if 'document' in request.json:
        documents = [request.json['document']]
        report = sketch.contains_from_text(documents, sort_chains_by_length=True)[0]
        report['width'] = SKETCH_WIDTH

        spans = []
        segments = []
        raw_segments = []

        raw_document = documents[0]
        processed_document = report['doc']
        clean_to_raw, raw_to_clean = tokenizations.get_alignments(processed_document, raw_document)
        for idx, chain in enumerate(report['chain_idxs']):
            chain_start, chain_end = chain[0], chain[-1] + report['width']
            # the alignment tables are character-wise: List[List[Int]] but the inner list is always of length 1
            raw_start = clean_to_raw[chain_start][0] # inc
            raw_end = clean_to_raw[chain_end - 1][-1] + 1 # inc

            spans.append((raw_start, raw_end))
            segments.append(''.join(report['chains'][idx]))
            raw_segments.append(raw_document[raw_start:raw_end])


        return jsonify({'spans' : spans, 'segments' : segments, 'raw_segments' : raw_segments})
    else:
        return jsonify("")

@app.route('/quip', methods=['POST'])
def process_documents():
    import dataportraits.span_utils
    from dataportraits.span_utils import latex_hl_start, latex_hl_end
    try:
        data = request.get_json()
        include_formatting = data.get('format_quotes', False)
        documents = data['documents']
        print(len(documents))
        with timers.Timer("Get Report") as t:
            reports = sketch.contains_from_text(documents, stride = 1, sort_chains_by_length=True)
            for report in reports:
                quip_report = {'too_short' : None, 'numerator' : None, 'denominator' : None, 'quip_25_beta': None}
                quip_report['numerator'] = sum(report['is_member'])
                quip_report['denominator'] = len(report['is_member'])
                if len(report['is_member']) == 0:
                    quip_report['too_short'] = True
                else:
                    quip_report['too_short'] = False
                    quip_report['quip_25_beta'] = quip_report['numerator'] / quip_report['denominator']
                report['quip_report'] = quip_report
                
                if include_formatting:
                    report['quote_latex'] = dataportraits.span_utils.format_chains(report, latex_hl_start, latex_hl_end)
                    report['quote_text'] = dataportraits.span_utils.format_chains(report, lambda num_quotes : "[" * num_quotes, lambda num_quotes : "]" * num_quotes)

            print(json.dumps(report, indent=2))
        return jsonify(reports)

    except Exception as e:
        print(e)
        return jsonify({"error": "Malformed request"}, 400)


if __name__ == '__main__':
    app.run(debug=False, port=8000, host='0.0.0.0', threaded=True)

