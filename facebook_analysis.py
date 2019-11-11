import pandas as pd
from bs4 import BeautifulSoup as bs
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys

def main(filepath):
    ### PREPROCESS
    def preprocess(filepath):
        page = open(filepath)
        soup = bs(page.read())    
        text = ''

        for message_raw in soup.find_all('div', {'class' : '_3-96 _2let'}):
            message = str(message_raw)

            def remove_tag(message, tag):
                delim_open = message.find('<' + tag)
                delim_close = message.find('</' + tag + '>')
                message = message.replace(message[delim_open : delim_close+(len(tag)+3)], '')
                return message  

            # Remove tags (using function)
            message = remove_tag(message, 'ul')
            message = remove_tag(message, 'video')

            # Remove http's (with regex)
            message = re.sub(r'http\S+', '', message)

            # Remove spefic strings (with replace)
            delim = '<div class="_3-96 _2let">'
            message = message.replace(delim, ' ')
            delim = 'X<br/>'
            message = message.replace(delim, ' ')
            delim = '<div>'
            message = message.replace(delim, '')
            delim = '</div>'
            message = message.replace(delim, '')
            delim_open = message.find('<a href=') # Position of delimiter
            delim_close = message.find('</a>')
            message = message.replace(message[delim_open : delim_close+4], '') # Remove all between delimiters

            text = text + ' ' + message
        # Save to file
        filename = NAME + '.txt'
        file = open(filename, 'w')
        file.write(text)

        return text
    
    ### SENTIMENT ANALYSIS
    def sentiment_analysis(text):    
        # Split raw text block into seperate tokens.
        def tokenize_text(text):
            pattern = r'\w+'
            tokenizer = RegexpTokenizer(pattern)
            text_tokenized = tokenizer.tokenize(text)
            return text_tokenized

        # Remove stop words.
        def remove_stopwords(text):
            # Combine multiple sets of stopwords. 
            stopwords_json = {"en":["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"]}
            stopwords_json_en = set(stopwords_json['en'])
            stopwords_nltk_en = set(stopwords.words('english'))
            stoplist_combined = set.union(stopwords_json_en, stopwords_nltk_en)
            rmsw = ([word for word in text if word not in stoplist_combined])
            return rmsw

        # Visualise Wordcloud 
        def wordcloud(text):
            # Remove neutral words
            sia = SentimentIntensityAnalyzer()
            comments = [comment for comment in text if (sia.polarity_scores(comment)['compound'] > 0) or (sia.polarity_scores(comment)['compound'] < 0)]

            # Find n-most common words
            word_frequency =  nltk.FreqDist(comments)
            text1 = [w[0]+':' + str(w[1]) for w in word_frequency.most_common(50)] 
            text2 = ' '.join(text1)

            # Plot wordcloud
            wordcloud = WordCloud(width=1600, height=800).generate(text2)
            plt.figure( figsize=(20,10), facecolor='k')
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(NAME + '-Wordcloud.png')
            #plt.show()

        def pie_chart(text):
            # Sort words by sentiment
            sia = SentimentIntensityAnalyzer()
            pos_comments = [comment for comment in text if sia.polarity_scores(comment)['compound'] > 0]
            neg_comments = [comment for comment in text if sia.polarity_scores(comment)['compound'] < 0]
            neu_comments = [comment for comment in text if comment not in pos_comments and comment not in neg_comments]

            # Plot pie chart
            labels = 'Positive', 'Negative'
            sizes = [len(pos_comments), len(neg_comments)]
            explode = (0, 0)  # Seperate slice from chart, example = 0.1
            colors = ['#99ff99','#ff9999']
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90, pctdistance=0.85)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            # Draw inner circle (aesthetic)
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            plt.savefig(NAME + '-PieChart.png')
    #         plt.show()
    #         plt.close()

        # Apply functions
        text_tokenized = tokenize_text(text)
        text_tokenized_rmsw = remove_stopwords(text_tokenized)
        pie_chart(text_tokenized_rmsw)
        wordcloud(text_tokenized_rmsw)
    
    text = preprocess(filepath)
    sentiment_analysis(text)

    
filepath = sys.argv[1] # Argument is filepath of FACEBOOK message.html in the form of /../facebook-***/messages/***_***/message.html (Important to include filepath inluding and onwards from messages/ )
NAME = re.search('/messages/(.*)_', filepath).group(1)

main(filepath)
