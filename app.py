import numpy as np
import pandas as pd
import json
from flask import Flask, request, render_template
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from difflib import SequenceMatcher

data=json.load(open("train-v2.0.json"))
x=data["data"]
z=x[3]['paragraphs']
q=len(z)
questions=[]
context=[]
nq=[0]
for i in range(q):
    #print(z[i]['qas'])
    x=len(z[i]['qas'])
    context.append(z[i]['context'])
    nq.append(x+nq[i])
    #print(x)
    for j in range(x):
    	questions.append(z[i]['qas'][j]['question'])
nq.remove(0)

model = BertForQuestionAnswering.from_pretrained('saved_model/')

tokenizer = BertTokenizer.from_pretrained('saved_tokenizer/')
from flask import Flask, request, render_template
accuracy=[]
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/",methods=["POST"])
def predict():
    if request.method=="POST":
        ques=request.form["question"]
        for i in range(len(questions)):
            accuracy.append(SequenceMatcher(None,questions[i],ques).ratio())
        idd=np.argmax(accuracy)
        c=0
        for i in range(len(nq)):
            if(idd<nq[i]):
                c=i
                break
        input_ids = tokenizer.encode(questions[idd],context[c])
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        sep_index = input_ids.index(tokenizer.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = len(input_ids) - num_seg_a
        segment_ids = [0]*num_seg_a + [1]*num_seg_b
        assert len(segment_ids) == len(input_ids)
        start_scores, end_scores = model(torch.tensor([input_ids]),token_type_ids=torch.tensor([segment_ids]))
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)
        answer = ' '.join(tokens[answer_start:answer_end+1])
        answer = tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            if tokens[i][0:2] == '##':
                answer += tokens[i][2:]
            else:
                answer += ' ' + tokens[i]
        ans='Answer: "' + answer + '"'
        '''c=0
        for i in range(len(nq)):
        if(idd<nq[i]):
            c=i
            break'''
    return render_template("index.html",y=ans)

if __name__ == "__main__":
    app.run()