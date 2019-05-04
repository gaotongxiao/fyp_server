from rasa_nlu.model import Metadata, Interpreter
import mysql.connector
import sys
import datetime


def get_tags(p):
    # where `model_directory points to the folder the model is persisted in`
    interpreter = Interpreter.load('models/smallset/nlu_model')
    result = interpreter.parse(p)
    # print(result['intent']['name'])

    # no prediction
    if result['intent']['name']=="android" and result['intent_ranking'][1]['name']=="swing":
        # print ("no tag")
        return "NAN"
    else: 
        print (result)
        return result['intent']['name']


def qid_query(input_tag, k=100):
    if input_tag=="NAN" or input_tag is None:
        return [[] for _ in range(4)]
    tags = input_tag.split('+')
    qid=[]
    qlist=[]
    tag_string = '"' 
    for tag in tags:
        tag_string += tag + ','
    tag_string = tag_string[:-1]
    tag_string += '"'
    print(tag_string)
    db = mysql.connector.connect(host='127.0.0.1', port=6000, user='root',password='fyp',db='fyp')
    cursor = db.cursor()
    t1 = datetime.datetime.now()
    cursor.execute('SELECT Question.Qid, Question.Title, Question.Content, Answer.Answer FROM Question INNER JOIN Answer ON Answer.Aid=Question.AcceptedAnswerId WHERE MATCH(Tags) AGAINST(%s) limit %d;' % (tag_string, k))
    results = cursor.fetchall()
    t2 = datetime.datetime.now()
    answers = []
    titles = []
    questions = []
    for row in results:
        titles.append(row[1])
        questions.append(row[1] + ' ' + row[2])
        answers.append(row[3])
        qid.append(row[0])
    print('------------------')
    print(t2-t1)
    return [qid, answers, titles, questions]


if __name__ == '__main__':
    a=get_tags(sys.argv[1])
    c, d = qid_query(a, 10000)



