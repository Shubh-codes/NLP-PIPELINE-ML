from flask import Flask, url_for, request, render_template, jsonify, send_from_directory
from flask_bootstrap import Bootstrap
import spacy
import time
from textblob import TextBlob

# Initialize the Spacy NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize Flask app
app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        start = time.time()
        
        # Processing the text
        rawtext = request.form['rawtext']
        docx = nlp(rawtext)
        custom_tokens = [token.text for token in docx]
        custom_wordinfo = [(token.text, token.lemma_, token.is_stop) for token in docx]
        lowered_text = rawtext.lower()
        length_text = str(len(rawtext))
        remove_punctuation = [token.text for token in docx if not token.is_punct]
        
        custom_postagging = [(word.text, word.tag_, word.pos_, word.dep_) for word in docx]
        custom_namedentities = [(entity.text, entity.label_) for entity in docx.ents]
        
        blob = TextBlob(rawtext)
        blob_sentiment, blob_subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity
        
        end = time.time()
        final_time = end - start
        
        return render_template('index.html', ctext=rawtext, custom_tokens=custom_tokens, length_text=length_text, remove_punctuation=remove_punctuation, custom_postagging=custom_postagging, custom_namedentities=custom_namedentities, custom_wordinfo=custom_wordinfo, lowered_text=lowered_text, blob_sentiment=blob_sentiment, blob_subjectivity=blob_subjectivity, final_time=final_time)
    else:
        return render_template('index.html')

@app.route('/basic_api')
def basic_api():
    return render_template('restfulapidocs.html')

@app.route('/imagescloud')
def imagescloud():
    # Serve the images cloud page
    return render_template('images.html')

@app.route('/about')
def about():
    # Serve the about page
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
