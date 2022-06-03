
from string import punctuation
from flask import Flask,url_for,request,render_template,jsonify,send_file
from flask_bootstrap import Bootstrap
import json
import nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import spacy
nltk.download('words')
from spacy import tokens
from textblob import TextBlob 
nlp = spacy.load("en_core_web_sm")

# WordCloud & Matplotlib Packages
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
from io import BytesIO
import random
import time


# Initialize App
app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/analyze',methods=['GET','POST'])
# We will lower case the enter data in the rawtext
def analyze():
	start = time.time()
	# Receives the input query from form
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		# Analysis
		docx = nlp(rawtext)
		# Tokens
		custom_tokens = [token.text for token in docx ]
		# Word Info
		custom_wordinfo = [(token.text,token.lemma_,token.is_stop) for token in docx ]
		lowered_text = rawtext.lower()
		length_text = str(len(rawtext))
		remove_punctuation = [token for token in docx if not token.is_punct]
		' '.join(token.text for token in remove_punctuation)
		
		remove_whitespace = rawtext.replace("   " , "").strip()
		


		custom_postagging = [(word.text,word.tag_,word.pos_,word.dep_) for word in docx]
		# NER
		custom_namedentities = [(entity.text,entity.label_)for entity in docx.ents]
		blob = TextBlob(rawtext)
		blob_sentiment,blob_subjectivity = blob.sentiment.polarity ,blob.sentiment.subjectivity
		# allData = ['Token:{},Tag:{},POS:{},Dependency:{},Lemma:{},Shape:{},Alpha:{},IsStopword:{}'.format(token.text,token.tag_,token.pos_,token.dep_,token.lemma_,token.shape_,token.is_alpha,token.is_stop) for token in docx ]
		allData = [('"Token":"{}","Tag":"{}","POS":"{}","Dependency":"{}","IsStopword":"{}"'.format(token.text,token.tag_,token.pos_,token.dep_,token.is_stop)) for token in docx ]

		result_json = json.dumps(allData, sort_keys = False, indent = 2)

		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,custom_tokens=custom_tokens,remove_whitespace = remove_whitespace ,length_text = length_text,remove_punctuation = remove_punctuation, custom_postagging=custom_postagging, custom_namedentities=custom_namedentities,custom_wordinfo=custom_wordinfo,lowered_text = lowered_text , blob_sentiment=blob_sentiment,blob_subjectivity=blob_subjectivity,final_time=final_time,result_json=result_json)




if __name__ == '__main__':
	app.run(debug=True)