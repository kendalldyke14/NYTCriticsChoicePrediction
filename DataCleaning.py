from ast import literal_eval
import re
import requests
from bs4 import BeautifulSoup
import time

def nyt_css_class(css_class):
    """
    This is a support function for the "get_nyt_review_text" function; it allows the BS4 parser to only extract desired text data.

    Args:
        css_class [bs4 class_ element]: This arg takes the css element passed to the keyword argument.

    Returns:
        boolean: Returns True if the argument matches, and False otherwise.
    """

    # Return relevant text fields:
    return css_class is not None and "css" in css_class

def get_nyt_review_text(url,re_cls=re.compile('class="(.*?)"')):
    """
    This function returns the text from the NYT movie review at the "url" input.

    Args:
        url [string]: The url for a NYT movie review. 
        re_cls [re.compile obj]: The compiled regex to determine the different css classes existing in a bs4 paragraph object.

    Returns:
        string: The text from a NYT movie review.
    """
    
    #Set up BeauifulSoup:
    session = requests.Session()
    session.headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.1.2222.33 Safari/537.36",
    "Accept-Encoding": "*",
    "Connection": "keep-alive"
    }
    req = session.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    paragraphs = soup.find_all('p', class_=nyt_css_class)
    #Convert paragraphs to list for processing:
    paragraphs_lst=list(paragraphs)
    #Get value of "class" for each separate "paragraph" object:
    paragraphs_cls=[re_cls.findall(str(cls))[0] for cls in paragraphs_lst]
    #The review text makes up the majority of the text on a review page, and is associated with the
    #class that has the most objects associated with it:
    cls_counts = dict()
    for i in paragraphs_cls:
        cls_counts[i] = cls_counts.get(i, 0) + 1
    review_cls = max(cls_counts, key=cls_counts.get)
    review_text=""
    for i in range(len(paragraphs)):
        if paragraphs_cls[i]==review_cls:
            review_text = review_text + paragraphs[i].get_text()
    time.sleep(4) #to reduce load
    return review_text

def get_omdb_awards(df_row, award_col="Awards"):
    """This function returns new, separated columns with integer values for the awards\
        a movie has won, from the "awards" column found in the OMDB API response data.

    Args:
        df_row (pandas DataFrame row): A single row from a pandas DataFrame, containing\
            a column with string data about the awards a particular movie has been \
                nominated for, or won.
        award_col (str): The name of the column containing the Awards string data.

    Returns:
        [int]: Six integers are returned, containing values for the number of oscar wins,\
            oscar nominations, emmy wins, emmy nominations, total wins, and total\
                nominations, respectively.
    """

    x=str(df_row[[award_col]][0])
    if "." in x:
        x=x.split(".")
        oscar_wins=[int(re.findall(r"\d+",x[0])[0]) if "Won" in x[0] and "Oscar" in x[0] else 0][0]
        oscar_noms=[int(re.findall(r"\d+",x[0])[0]) if "Nominated" in x[0] and "Oscar" in x[0] else 0][0]
        emmy_wins=[int(re.findall(r"\d+",x[0])[0]) if "Won" in x[0] and "Emmy" in x[0] else 0][0]
        emmy_noms=[int(re.findall(r"\d+",x[0])[0]) if "Nominated" in x[0] and "Emmy" in x[0] else 0][0]
        total_wins=[int(re.findall(r"\d+(?= win)",x[1])[0]) if "win" in x[1] else 0][0]
        total_noms=[int(re.findall(r"\d+(?= nomination)",x[1])[0]) if "nomination" in x[1] else 0][0]
    else:
        oscar_wins, oscar_noms, emmy_wins, emmy_noms=0,0,0,0
        total_wins=[int(re.findall(r"\d+(?= win)",x)[0]) if "win" in x else 0][0]
        total_noms=[int(re.findall(r"\d+(?= nomination)",x)[0]) if "nomination" in x else 0][0]
    return oscar_wins,oscar_noms,emmy_wins,emmy_noms,total_wins,total_noms