import json
from nltk.util import ngrams
import metric

f=open('test_ppl.json')
lines=json.load(f)
c=lines[4]
multi_ref=lines[2]
keyword=lines[1]
query=lines[0]
expose=lines[7]
keyword_ng=[]
for i in range(len(query)):
    qk=list(set(query[i]+keyword[i]))
    keyword_ng.append(qk)
log_path='./'

bleu, recall_q, recall_k, recall_qk, dist1, dist2=\
    metric.get_metric(c, multi_ref, keyword, query, keyword_ng,expose, log_path)
print(' bleu{}, recall_q{}, recall_k{}, recall_qk{}, dist1{}, dist2{} \n'.format(bleu, recall_q, recall_k, recall_qk, dist1, dist2))