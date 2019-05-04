# this file is for automatically correction of the similairty score in UserEval table
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

print("mysql connecting...")
conn = mysql.connector.connect(port=6000, user='root', password='fyp', database='fyp', auth_plugin='mysql_native_password')
app = flask.Flask(__name__)
cors = CORS(app, resources={r"/predict": {"origins": "*"}, r"/eval": {"origins": "*"}, r"/voice": {"origins": "*"}})
TEXT = dill.load(open("models/torchtext.words", "rb"))
db_cursor = conn.cursor()
print("model running...")
m = Comparer(TEXT.vocab).cuda()
m.load_state_dict(torch.load('models/model.pt', map_location='cuda'))
m.eval()
print("sphinx connecting...")
sphinx = SphinxClient()
sphinx.SetLimits(0, k, k)
sphinx.SetRankingMode(1)
sphinx.SetConnectTimeout(0.5)

def preprocess(s):
    global TEXT
    s = re.sub(r'<code>.*?</code>', '', s, flags=re.DOTALL)
    s = re.sub('<.*?>', ' ', s)
    translator = s.maketrans(string.punctuation, ' '*len(string.punctuation))
    s = s.translate(translator)
    s = re.sub(r'\n+', ' ', s)
    return TEXT.preprocess(s)

def sphinx_match(s):
    global sphinx, db_cursor
    s = ' '.join(preprocess(s))
    print(s)
    #sphinx = pymysql.connect(port=9312, user='root', password='fyp')
    #sphinx_cursor = sphinx.cursor()
    #sphinx_query = "select id from test1 where match('%s') limit %s" % (s, str(k))
    #sphinx_cursor.execute(sphinx_query)
    #sphinx_res = sphinx_cursor.fetchall()
    #sphinx.close()
    res = sphinx.Query(s, 'test1')
    ids = ''
    mid = 10000000
    if len(res['matches']) == 0:
        return [], [], []
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
    for cand in result:
        cands.append(cand[2])
        qids += str(cand[0]) + ','
    qids = qids[:-1]
    db_cursor.execute('SELECT Title, Content FROM Question WHERE Qid in (%s) ORDER BY FIELD (Qid,%s);' % (qids, qids))
    ques_cands = db_cursor.fetchall()
    titles = []
    questions = []
    for c in ques_cands:
        titles.append(c[0])
        questions.append(c[0] + ' ' + c[1])
    #ques_cands = [c[0] for c in ques_cands]
    return cands, titles, questions


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
        

def update(id, score):
    q = 'UPDATE `UserEval` SET sim_score=%s WHERE id=%s'
    db_cursor.execute(q, (str(score), str(id)))
    conn.commit()

def predict(input_ques=None):
    global m, TEXT, BATCHSIZE, k, db_cursor, conn
    if input_ques is None:
        r_json = request.json
        input_ques = r_json['data']
    pp_input_ques = preprocess(input_ques)
    x, x_len = TEXT.process([pp_input_ques])
    x = x.cuda()
    top_word_score, top_word_idx = map(lambda x: x.detach(), m.encoder(x, True))
    sphinx_query = set()
    posTagged = pos_tag(nltk.word_tokenize(input_ques))
    simplifiedTags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in posTagged]
    for word, tag in simplifiedTags:
        if (tag == 'VERB' or tag == 'NOUN') and word.lower() not in stopwords.words('english'):
            sphinx_query.add(word.lower())
    for i, e in enumerate(top_word_idx.cpu().numpy()[0]):
        if top_word_score[0][i] < 1 / len(pp_input_ques) or e >= len(pp_input_ques): 
            continue
        sphinx_query.add(pp_input_ques[e].lower())
    sphinx_query = ' '.join(sphinx_query)
    print(sphinx_query)
    sphinx_cands, sphinx_titles, sphinx_questions = sphinx_match(sphinx_query)
    t = datetime.datetime.now()
    tag_cands = get_tags(input_ques)
    t2 = datetime.datetime.now()
    print("tag time:", t2-t)
    print("tags", tag_cands)
    tag_cands, tag_titles, tag_questions = qid_query(tag_cands, k)
    t3 = datetime.datetime.now()
    print("qid time:", t3-t2)

    titles = sphinx_titles + tag_titles
    questions = sphinx_questions + tag_questions
    cands = sphinx_cands + tag_cands


    y1, _ = TEXT.process([preprocess(t) for t in titles])
    y2, _ = TEXT.process([preprocess(c) for c in questions])
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
    top_scores, top_idxs = map(lambda x: x.cpu().numpy(), overall_score[:, 1].topk(1, 0))
    best_p = questions[top_idxs[0]]
    best_a = cands[top_idxs[0]]
    return top_scores[0]

if __name__ == '__main__':
    q = 'select id, user_question from UserEval'
    db_cursor.execute(q)
    res = db_cursor.fetchall()
    for i in res:
        update(i[0], predict(i[1]))

