import multiprocessing as mp
import math
import string
import json
import numpy as np
import flask
from flask import Flask, request, Response, render_template
from flask_cors import CORS
from torchtext import data
import dill
import torch
import re
import ast
import nltk
from nltk.tag import pos_tag, map_tag
from nltk.corpus import stopwords
from model import Comparer
import mysql.connector
import pymysql
from sphinxapi.sphinxapi import *
import datetime
from load_nlu_model import get_tags, qid_query

k = 1000 # how many candidates
BATCHSIZE = 256
USE_CUDA = 1

print("USE_CUDA:", USE_CUDA)
print("mysql connecting...")
conn = mysql.connector.connect(port=6000, user='root', password='fyp', database='fyp', auth_plugin='mysql_native_password')
app = flask.Flask(__name__)
cors = CORS(app, resources={r"/predict": {"origins": "*"}, r"/eval": {"origins": "*"}, r"/TA": {"origins": "*"}})
TEXT = dill.load(open("models/torchtext.words", "rb"))
db_cursor = conn.cursor()
print("model running...")
if USE_CUDA:
    m = Comparer(TEXT.vocab).cuda()
    m.load_state_dict(torch.load('models/model.pt', map_location='cuda'))
else:
    m = Comparer(TEXT.vocab)
    m.load_state_dict(torch.load('models/model.pt', map_location='cpu'))
m.eval()
print("sphinx connecting...")
sphinx = SphinxClient()
sphinx.SetLimits(0, k, k)
sphinx.SetRankingMode(1)
sphinx.SetConnectTimeout(10.0)

def preprocess(s):
    global TEXT
    s = re.sub(r'<code>.*?</code>', '', s, flags=re.DOTALL)
    s = re.sub('<.*?>', ' ', s)
    translator = s.maketrans(string.punctuation, ' '*len(string.punctuation))
    s = s.translate(translator)
    s = re.sub(r'\n+', ' ', s)
    return TEXT.preprocess(s)

def sphinx_match(s):
    s = ' '.join(preprocess(s))
    print(s)
    res = sphinx.Query(s, 'test1')
    ids = ''
    mid = 10000000
    if len(res['matches']) == 0:
        return [[] for _ in range(4)]
    for idx_res in res['matches']:
        ids += str(idx_res['id']) + ','
    '''
    if len(sphinx_res) == 0:
        return [], [], []
    for idx_res in sphinx_res:
        ids += str(idx_res[0]) + ','
    '''
    ids = ids[:-1]
    t = datetime.datetime.now()
    db_cursor.execute('select * from Answer where Aid in (%s)' % (ids))
    result = db_cursor.fetchall()
    print(len(result))
    print("---------------------")
    t2 = datetime.datetime.now()
    print('database time:', t2-t)
    cands = []
    qids = ''
    ques_cands = []
    qid_list = []
    for cand in result:
        cands.append(cand[2])
        qids += str(cand[0]) + ','
        qid_list.append(cand[0])
    qids = qids[:-1]
    db_cursor.execute('SELECT Title, Content FROM Question WHERE Qid in (%s) ORDER BY FIELD (Qid,%s);' % (qids, qids))
    ques_cands = db_cursor.fetchall()
    titles = []
    questions = []
    for c in ques_cands:
        titles.append(c[0])
        questions.append(c[0] + ' ' + c[1])
    #ques_cands = [c[0] for c in ques_cands]
    return [qid_list, cands, titles, questions]

def tag_match(input_ques):
    tag_cands = get_tags(input_ques)
    return qid_query(tag_cands, k)

def passQA_match():
    db_cursor.execute('select * from TAData')
    ques_cands = db_cursor.fetchall()
    questions = []
    answers = []
    ids = [] # useless
    for c in ques_cands:
        ids.append(c[0])
        questions.append(c[1])
        answers.append(c[2])
    return [ids, answers, questions, questions]


def batch_feed(x, y, batchsize):
    # x: single sample
    # y: batch of samples
    loop_time = (len(y) + batchsize - 1) // batchsize
    
 
class torchIter:
    def __init__(self, batchsize, x, *args):
        self.samples = args[0].shape[1]
        self.batchsize = batchsize
        self.ctimes = (self.samples + batchsize - 1) // batchsize
        self.params = args
        self.x = x
        
    def __iter__(self):
        self.ctr = 0
        return self
    
    def __next__(self):
        if self.ctr >= self.ctimes:
            raise StopIteration
        st = self.batchsize * self.ctr
        ed = st + self.batchsize
        self.ctr += 1
        x_rep_times = self.batchsize if self.samples - st >= self.batchsize \
                        else self.samples - st
        retList = [self.x.repeat(1, x_rep_times)]
        retList += [p[:, st:ed] for p in self.params]
        return tuple(retList) 
        

@app.route("/eval", methods=['POST'])
def eval():
    print("eval")
    r_json = request.json
    if int(r_json['user_score']) not in [1, 2, 3]:
        response = json.dumps({'status' : 0})
        return Response(response=response, status=200, mimetype="application/json")
    q = 'INSERT INTO `UserEval` (user_question, Question, Answer, sim_score, user_score) VALUE (%s, %s, %s, %s, %s)'
    db_cursor.execute(q, (r_json['user_question'], r_json['Question'], r_json['Answer'], str(float(r_json['sim_score'])), r_json['user_score']))
    conn.commit()
    return Response(response=json.dumps({'status' : 1}), status=200, mimetype="application/json")

@app.route("/TA", methods=['POST'])
def TA():
    print("TA")
    r_json = request.json
    q = 'INSERT INTO `TAData` (Question, Answer) VALUE (%s, %s)'
    print(r_json)
    db_cursor.execute(q, (r_json['Question'], r_json['Answer']))
    conn.commit()
    return Response(response=json.dumps({'status' : 1}), status=200, mimetype="application/json")

'''
@app.route("/voice", methods=['POST'])
def voice():
    print("get voice")
    content = request.get_data()
    print(content)
    print("===========")
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    from google.oauth2 import service_account
    
    credentials = service_account.Credentials.from_service_account_file('/home/data/dy_fyp/fyp/fyp-speech-recognition-5c8ecddfdd42.json')

    client = speech.SpeechClient(credentials=credentials)
    #not sure about the type, need to double check
    #content = request.json
    audio = types.RecognitionAudio(content=content)
    config = types.RecognitionConfig(encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,sample_rate_hertz=16000, language_code='en-US')
    #Detects speech in the audio file
    response = client.recognize(config, audio)
    for result in response.results:
        r_json=result.alternatives[0].transcript
        print(r_json)
    #r_json=ast.literal_eval(r_json)
    #print(r_json)
'''

@app.route("/predict", methods=['POST'])
def predict(input_ques=None):
    global m, TEXT, BATCHSIZE, k, db_cursor, conn
    if input_ques is None:
        r_json = request.json
        input_ques = r_json['data']
    pp_input_ques = preprocess(input_ques)
    x, x_len = TEXT.process([pp_input_ques])
    if USE_CUDA:
        x = x.cuda()
    top_word_score, top_word_idx = map(lambda x: x.detach(), m.encoder(x, True))
    sphinx_query = set()
    posTagged = pos_tag(nltk.word_tokenize(input_ques))
    simplifiedTags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in posTagged]
    for word, tag in simplifiedTags:
        if (tag == 'VERB' or tag == 'NOUN') and word.lower() not in stopwords.words('english'):
            sphinx_query.add(word.lower())
    print("HIHIHI", sphinx_query)
    for i, e in enumerate(top_word_idx.cpu().numpy()[0]):
        if top_word_score[0][i] < 1 / len(pp_input_ques) or e >= len(pp_input_ques): 
            continue
        sphinx_query.add(pp_input_ques[e].lower())
    sphinx_query = ' '.join(sphinx_query)
    print(sphinx_query)

    # MultiProcessing Prepare
    mp_res = mp.Queue(maxsize=0)
    ress = [sphinx_match(sphinx_query), tag_match(input_ques), passQA_match()]
    '''
    p1 = mp.Process(target=sphinx_match, args=(sphinx_query, mp_res,))
    p2 = mp.Process(target=tag_match, args=(input_ques, mp_res,))
    p3 = mp.Process(target=passQA_match, args=(mp_res,))
    p1.start()
    p1.join()
    print("hihi")
    p2.start()
    p3.start()
    p1.join()
    print('p1 end')
    p2.join()
    print('p1 end')
    p3.join()
    print('p1 end')
    '''
    titles = []
    questions = []
    cands = []
    qids = []
    for res in ress:
        qids += res[0]
        cands += res[1]
        titles += res[2]
        questions += res[3]
        

    '''
    #t = datetime.datetime.now()
    tag_cands = get_tags(input_ques)
    #t2 = datetime.datetime.now()
    #print("tag time:", t2-t)
    #print("tags", tag_cands)
    tag_qids, tag_cands, tag_titles, tag_questions = qid_query(tag_cands, k)
    #t3 = datetime.datetime.now()
    #print("qid time:", t3-t2)

    # TA cands
    TA_ids, TA_ques, TA_cands = pass_QA_fetch()

    titles = sphinx_titles + tag_titles + TA_ques
    questions = sphinx_questions + tag_questions + TA_ques
    cands = sphinx_cands + tag_cands + TA_cands
    qids = sphinx_qids + tag_qids + TA_ids
    '''


    y1, _ = TEXT.process([preprocess(t) for t in titles])
    y2, _ = TEXT.process([preprocess(c) for c in questions])
    if USE_CUDA:
        y1, y2 = y1.cuda(), y2.cuda()
    d = torchIter(BATCHSIZE, x, y1, y2)
    agm = 0
    overall_score = None
    for i, (bx, by1, by2) in enumerate(d):
        #score_title = m(bx, by1).detach()
        score_content = m(bx, by2).detach()
        score = score_content
        overall_score = score if overall_score is None else torch.cat([overall_score, score], 0)

    overall_score = torch.nn.functional.softmax(overall_score, 1)
    top_scores, top_idxs = map(lambda x: x.cpu().numpy(), overall_score[:, 1].topk(5, 0))
    #best_p = questions[top_idxs[0]]
    #best_a = cands[top_idxs[0]]
    best_p = []
    best_a = []
    best_title = []
    return_qids = []
    for i, idx in enumerate(top_idxs):
        return_qids.append(qids[idx])
        best_p.append(questions[idx])
        best_a.append(cands[idx])
        best_title.append(titles[idx])
    response = {'agm': top_scores[0].tolist(), 'best_p': best_p, 'best_a': best_a, 'ids': return_qids, 'best_title': best_title}
    response_pickled = json.dumps(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == '__main__':
    app.run(port=8181)
