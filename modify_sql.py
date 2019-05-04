# this is written to convert tags from like "<A><B>" to "A,B"
import mysql.connector

conn = mysql.connector.connect(port=6000, user='root', password='fyp', database='fyp', auth_plugin='mysql_native_password')
db_cursor = conn.cursor()
db_cursor.execute('select Qid, Tags from Question')
Qids = db_cursor.fetchall()
i = 0
for Qid, Tags in Qids:
    if i == 5000:
        qid_values = qid_values[:-1]
        db_cursor.execute('UPDATE Question SET Tags = CASE Qid %s END WHERE Qid in (%s)' % (values, qid_values))
        conn.commit()
        i = 0
    if Tags.find('<')!=-1 and Tags.find('>')!=-1:
        ftags = Tags.replace('<', '')
        ftags = ftags.replace('>', ',')[:-1]
    else:
        ftags = Tags;
    if i == 0:
        values = ''
        qid_values = ''
    values += 'WHEN %s THEN "%s" ' % (Qid, ftags)
    qid_values += str(Qid) + ','
    i += 1
qid_values = qid_values[:-1]
db_cursor.execute('UPDATE Question SET Tags = CASE Qid %s END WHERE Qid in (%s)' % (values, qid_values))
conn.commit()
