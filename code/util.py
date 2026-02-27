import pyphen
import re
from langdetect import detect_langs
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from janome.tokenizer import Tokenizer
import spacy
from typing import Tuple


# Initialize the tokenizer
dic_en = pyphen.Pyphen(lang='en')  # English
dic_fr = pyphen.Pyphen(lang='fr')  # Franch
dic_de = pyphen.Pyphen(lang='de')  # German
dic_ru = pyphen.Pyphen(lang='ru')  # Russian



################Functions needed to generate rule-based constraints#################
def robust_detect_lang(text):
    """
    Use langdetect to return the language code with the highest probability,
    make a special judgment for Chinese.
    """
    try:
        langs = detect_langs(text)
        top_lang = langs[0].lang
        if top_lang == 'ko':
            #  If it is ko, but contains a large number of Chinese characters, we forcibly judge it as Chinese.
            zh_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
            if zh_count / len(text) > 0.5:
                return 'zh'
        return top_lang
    except Exception:
        return 'unknown'

def normalize_lang_code(lang_code):
    """
    Return the generic language prefix, such as zh-cn → zh, en-us → en.
    """
    return lang_code.lower().split('-')[0]

### number of words

def count_words(text, language):
    """
    Parameters:
        text: The string to be counted.
        language: Language code ('zh' - Chinese, 'en' - English, 'fr' - French,
                 'ru' - Russian, 'ja' - Japanese, 'de' - German)

    Returns:
        The number of characters or words.
    """
    if not text.strip():
        return 0

    if language == 'zh':
        # # Use regular expressions to remove punctuation and spaces
        cleaned_text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
        return len(cleaned_text)
    elif language == 'ja':
        t = Tokenizer()
        tokens = list(t.tokenize(text))
        return len(tokens)
    else:
        # # Choose the appropriate tokenization method based on the language
        if language == 'en':
            words = word_tokenize(text, language='english')
        elif language == 'fr':
            words = word_tokenize(text, language='french')
        elif language == 'de':
            words = word_tokenize(text, language='german')
        elif language == 'ru':
            words = word_tokenize(text, language='russian')
        else:
            words = word_tokenize(text)

        # # Filter out punctuation marks
        words = [word for word in words if re.match(r'^\w+$', word)]
        return len(words)

def nearest_hundreds(num):
    lower = (num // 100) * 100
    upper = lower + 100
    return lower, upper
def nearest_ten(num):
    lower = (num // 10) * 10
    return lower


### number of sentence


# # Load the spaCy model (the corresponding language package needs to be installed in advance)
nlp_fr = spacy.load("fr_core_news_sm") if spacy.util.is_package("fr_core_news_sm") else None
nlp_de = spacy.load("de_core_news_sm") if spacy.util.is_package("de_core_news_sm") else None
nlp_ru = spacy.load("ru_core_news_sm") if spacy.util.is_package("ru_core_news_sm") else None

def count_sentences(text, language):
    """
    Count the number of sentences in a multilingual text.

    Parameters:
        text: Input text string
        language: Language code ('zh', 'en', 'fr', 'ru', 'ja', 'de')

    Returns:
        Number of sentences (int)
    """
    if not text.strip():
        return 0

    text = text.replace('\n', ' ')

    if language == 'zh':
        sentences = re.split(r'[。！？；…]+', text)
        return len([s for s in sentences if s.strip()])

    elif language == 'ja':
        sentences = re.split(r'[。！？・…]+', text)
        return len([s for s in sentences if s.strip()])

    elif language == 'en':
        return len(sent_tokenize(text, language='english'))

    elif language == 'fr' and nlp_fr:
        doc = nlp_fr(text)
        return len(list(doc.sents))

    elif language == 'de' and nlp_de:
        doc = nlp_de(text)
        return len(list(doc.sents))

    elif language == 'ru' and nlp_ru:
        doc = nlp_ru(text)
        return len(list(doc.sents))

    else:
        return len(sent_tokenize(text))


# format : json markdown xml html

def detect_text_format(text):
    """
    Parameters:
        text: Input text string

    Returns:
        0: No special format (plain text)
        1: JSON format
        2: Markdown format
        3: XML format
        4: HTML format
    """
    text = text.strip()
    if not text:
        return 0

    if detect_json(text):
        return 1

    if detect_markdown(text):
        return 2

    xml_html_type = detect_xml_html(text)
    if xml_html_type == "xml":
        return 3
    elif xml_html_type == "html":
        return 4

    return 0


def detect_json(text):
    json_pattern = r'^\s*(\{.*\}|\[.*\])\s*$'
    json_partial_pattern = r'(\{.*\}|\[.*\])'

    if re.fullmatch(json_pattern, text, re.DOTALL):
        try:
            import json
            json.loads(text)
            return True
        except:
            pass
    elif re.search(json_partial_pattern, text, re.DOTALL):
        try:
            import json
            json_str = re.search(json_partial_pattern, text, re.DOTALL).group(1)
            json.loads(json_str)
            return True
        except:
            pass
    return False


def detect_markdown(text):
    patterns = [
        r'^#\s+.+', r'^[*+-]\s+.+', r'^\d+\.\s+.+',
        r'!?\[.*\]\(.+\)', r'^\s*`{3}[\s\S]*?`{3}',
        r'^\s*\|.+\|', r'^\s*>+.+'
    ]
    return any(re.search(p, text, re.MULTILINE) for p in patterns)


def detect_xml_html(text):
    """
    XML/HTML Format Detection and Differentiation
    Return: "xml" | "html" | None
    """

    xml_declaration = r'<\?xml\s+version='
    xml_tag = r'<([a-z][a-z0-9]*)(?:\s+[^>]*)?>.*?</\1>'

    html5_self_closing = r'<(img|br|hr|input|meta|link|source|track|embed|wbr)\b[^>]*/?>'
    html5_attrs = r'\s(data-|aria-)[a-z-]+='

    html_doctype = r'<!DOCTYPE\s+html>'
    html_tags = r'<(html|head|body|div|span|a|p)\b'
    html_self_closing = r'<(br|hr|img|input|meta|link)\b[^>]*>'
    html_attr = r'\s(id|class|style)=["\'][^"\']*["\']'

    if re.search(xml_declaration, text, re.IGNORECASE):
        return "xml"
    if re.search(html_doctype, text, re.IGNORECASE):
        return "html"

    xml_score = 0
    html_score = 0

    if re.search(xml_tag, text, re.DOTALL | re.IGNORECASE):
        xml_score += 2
    if '<?xml' in text.lower():
        xml_score += 3

    if re.search(html5_self_closing, text, re.IGNORECASE):
        html_score += 2
    if re.search(html5_attrs, text, re.IGNORECASE):
        html_score += 1

    if re.search(html_tags, text, re.IGNORECASE):
        html_score += 2
    if re.search(html_self_closing, text, re.IGNORECASE):
        html_score += 1
    if re.search(html_attr, text):
        html_score += 1
    if '<html' in text.lower():
        html_score += 3

    if xml_score > html_score and xml_score >= 2:
        return "xml"
    elif html_score > xml_score and html_score >= 2:
        return "html"
    elif xml_score == html_score and xml_score >= 2:
        # # When the feature scores are tied, prioritize judging it as HTML (since HTML is more common)
        return "html"

    return None

### keyword

def count_substring(main_str, sub_str):
    """
    Parameters:
        main_str: The main string
        sub_str: The substring to be searched for

    Returns:
        Number of occurrences (int)
    """
    if not sub_str:
        return 0
    return main_str.count(sub_str)


### start end with

def extract_first_last_words(text: str) -> Tuple[str, str]:
    """
    Parameters:
        text: Input text string (supports Chinese, English, French, Russian, German, Japanese, etc.)

    Returns:
        (first_word, last_word): A tuple containing the first and last words.
        If the text contains no valid words, returns ("", "").
    """
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text.strip())
    if not text:
        return "", ""

    # Improved tokenization logic:
    # 1. Match all Unicode letters, digits, kana, Chinese characters, Cyrillic letters, etc.
    # 2. Include connectors (such as the apostrophe in French, like "l'apostrophe")
    words = re.findall(
        r'[\w\u0400-\u04FF\u00C0-\u017F\u3040-\u30FF\u4E00-\u9FFF]+(?:[\'-][\w\u0400-\u04FF\u00C0-\u017F\u3040-\u30FF\u4E00-\u9FFF]+)*',
        text,
        re.UNICODE
    )

    if not words:
        return "", ""

    first_word = words[0]
    last_word = words[-1]

    # Expand the range of punctuation cleaning (supports punctuation in multiple languages)
    punctuation = r'''.!?,;:'"‘’“”«»„‟()[]{}<>~`@#$%^&*_\-+=|\\/…¿¡°•·'''
    first_word = first_word.strip(punctuation)
    last_word = last_word.strip(punctuation)

    return first_word, last_word


###uppercase and lowercase

def check_english_uppercase(text):
    """
    Parameters:
        text: Input text string

    Returns:
        0: Not English or English but not all uppercase
        1: All uppercase English
    """
    if not text.strip():
        return 0

    # Detect whether it is English (excluding cases containing non-English characters)
    if not re.fullmatch(r'^[a-zA-Z\s\.,!?;:\'"-]+$', text):
        return 0

    # Extract all alphabetic characters (excluding punctuation, spaces, etc.)
    letters = re.findall(r'[a-zA-Z]', text)
    if not letters:  # 无字母情况
        return 0

    # Determine if all characters are uppercase
    return 1 if all(c.isupper() for c in letters) else 0


def check_english_lowercase(text):
    """
    Parameters:
        text: Input text string

    Returns:
        0: Not English or English but not all uppercase
        1: All uppercase English
    """
    if not text.strip():
        return False

    # Detect whether it is English (excluding cases containing non-English characters)
    if not re.fullmatch(r'^[a-zA-Z\s\.,!?;:\'"-]+$', text):
        return False

    # Extract all alphabetic characters (excluding punctuation, spaces, etc.)
    letters = re.findall(r'[a-zA-Z]', text)
    if not letters:  # 无字母情况
        return False

    # Determine if all characters are lowercase
    return True if all(c.islower() for c in letters) else False

###No Commas

def contains_no_punctuation(text, punctuation_chars={'，', ',', '﹐',} ):
    return not any(punc in text for punc in punctuation_chars)



################Functions that will be used in the evaluation#################
def evaluate_start_with(response,strat_with_word):
    first_word, _ = extract_first_last_words(response)
    return first_word == strat_with_word
def evaluate_end_with(response, end_with_word):
    _, last_word = extract_first_last_words(response)
    return end_with_word == last_word
def evaluate_keyword(response, keyword, num):
    count_num = count_substring(response, keyword)
    return count_num == num

def evaluate_format(response, format_type):
    # format: JSON\MARKDOWN\XML\HTML
    format_id = detect_text_format(response)
    id2type = {0: "NO_FORMAT", 1:"JSON", 2:"MARKDOWN", 3:"XML", 4:"HTML"}
    return id2type[format_id] == format_type

def evaluate_word_length(response, word_num_bottom_top,around_word_num,word_length_bar, word_length_template_type):
    lang_full = robust_detect_lang(response)
    lang_short = normalize_lang_code(lang_full)
    word_num = count_words(response, lang_short)
    if word_length_template_type == 0:
        # Around the target word count, with a 20% fluctuation, use around_word_num as the target word count
        low = int(around_word_num * 0.8)
        high = int(around_word_num * 1.2)
        return word_num >= low and word_num <= high
    elif word_length_template_type == 1:
        # Below is lower than the bar, using word_length_bar as the bar
        return word_num <= word_length_bar
    return word_num >= word_num_bottom_top[0] and word_num <= word_num_bottom_top[1]

def evaluate_sentence_length(response, sentence_length_template_type,sentence_length_target,sentence_length_bottom_top):
    lang_full = robust_detect_lang(response)
    lang_short = normalize_lang_code(lang_full)
    sentence_num = count_sentences(response, lang_short)
    if sentence_length_template_type == 0:
        # exactly
        return sentence_num == sentence_length_target
    elif sentence_length_template_type == 1:
        # around
        return sentence_num >= max(sentence_length_target-2, 0) and sentence_num <= sentence_length_target +2
    elif sentence_length_template_type == 2:
        # below num of sentence + 2
        return sentence_num <= sentence_length_target + 2

    # between
    return sentence_num >= sentence_length_bottom_top[0] and sentence_num <= sentence_length_bottom_top[1]




class MyClass:
    def evaluate_word_length(response, word_num_bottom_top, around_word_num, word_length_bar,
                             word_length_template_type):
        lang_full = robust_detect_lang(response)
        lang_short = normalize_lang_code(lang_full)
        word_num = count_words(response, lang_short)
        if word_length_template_type == 0:
            # Around the target word count, with a 20% fluctuation, use around_word_num as the target word count
            low = int(around_word_num * 0.8)
            high = int(around_word_num * 1.2)
            return word_num >= low and word_num <= high
        elif word_length_template_type == 1:
            # Below is lower than the bar, using word_length_bar as the bar
            return word_num <= word_length_bar
        return word_num >= word_num_bottom_top[0] and word_num <= word_num_bottom_top[1]

    def evaluate_sentence_length(response, sentence_length_template_type, sentence_length_target,
                                 sentence_length_bottom_top):
        lang_full = robust_detect_lang(response)
        lang_short = normalize_lang_code(lang_full)
        sentence_num = count_sentences(response, lang_short)
        if sentence_length_template_type == 0:
            # exactly
            return sentence_num == sentence_length_target
        elif sentence_length_template_type == 1:
            # around
            return sentence_num >= max(sentence_length_target - 2, 0) and sentence_num <= sentence_length_target + 2
        elif sentence_length_template_type == 2:
            # below num of sentence + 2
            return sentence_num <= sentence_length_target + 2

        # between
        return sentence_num >= sentence_length_bottom_top[0] and sentence_num <= sentence_length_bottom_top[1]

    def evaluate_keyword(response, keyword, num):
        count_num = count_substring(response, keyword)
        return count_num == num

    def evaluate_format(response, format_type):
        # format: JSON\MARKDOWN\XML\HTML
        format_id = detect_text_format(response)
        id2type = {0: "NO_FORMAT", 1: "JSON", 2: "MARKDOWN", 3: "XML", 4: "HTML"}
        return id2type[format_id] == format_type

    def evaluate_start_with(response, strat_with_word):
        first_word, _ = extract_first_last_words(response)
        return first_word == strat_with_word

    def evaluate_end_with(response, end_with_word):
        _, last_word = extract_first_last_words(response)
        return end_with_word == last_word

    ###uppercase and Lowercase

    def check_english_uppercase(text):
        """
        Parameters:
            text: Input text string

        Returns:
            0: Not English or English but not all uppercase
            1: All uppercase English
        """
        if not text.strip():
            return 0

        # Detect whether it is English (excluding cases containing non-English characters)
        if not re.fullmatch(r'^[a-zA-Z\s\.,!?;:\'"-]+$', text):
            return 0

        # Extract all alphabetic characters (excluding punctuation, spaces, etc.)
        letters = re.findall(r'[a-zA-Z]', text)
        if not letters:
            return 0

        # Determine if all characters are uppercase
        return 1 if all(c.isupper() for c in letters) else 0

    def check_english_lowercase(text):
        """
        Detect whether the text is all uppercase English.
        Parameters:
            text: Input text string

        Returns:
            0: Not English or English but not all uppercase
            1: All uppercase English
        """
        if not text.strip():
            return False

        # Detect whether it is English (excluding cases containing non-English characters)
        if not re.fullmatch(r'^[a-zA-Z\s\.,!?;:\'"-]+$', text):
            return False

        # Extract all alphabetic characters (excluding punctuation, spaces, etc.)
        letters = re.findall(r'[a-zA-Z]', text)
        if not letters:
            return False

        # Determine if all characters are lowercase
        return True if all(c.islower() for c in letters) else False

    ###No Commas

    def contains_no_punctuation(text, punctuation_chars={'，', ',', '﹐', }):
        return not any(punc in text for punc in punctuation_chars)
