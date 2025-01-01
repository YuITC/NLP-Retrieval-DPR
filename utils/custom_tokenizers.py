import copy
import regex
import spacy

from utils.dpr_utils import get_logger
logger = get_logger(__file__) # Customized logger

class Tokens(object):
    TEXT, TEXT_WS, SPAN = 0, 1, 2
    POS , LEMMA  , NER  = 3, 4, 5

    def __init__(self, data, annotators, opts=None):
        """
        Represents a list of tokenized text with associated annotations.

        Args:
            data      (list): List of tokens, each token is a tuple contains (TEXT, TEXT_WS, SPAN, [POS, LEMMA, NER]).
            annotators (set): Set of active annotators (e.g., "pos", "lemma", "ner").
            opts      (dict): Optional configuration options.    
        """
        self.data       = data
        self.annotators = annotators
        self.opts       = opts or {}

    def __len__(self):
        """
        The number of tokens.
        """
        return len(self.data)

    def slice(self, i=None, j=None):
        """
        Return Tokens object containing tokens from [i, j).
        """
        new_tokens      = copy.copy(self)
        new_tokens.data = self.data[i:j]
        return new_tokens

    def untokenize(self):
        """
        Return original text with whitespace.
        """
        return "".join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """
        Return a list of token texts.

        Args:
            uncased: Lowercases text.
        """
        return [t[self.TEXT].lower() for t in self.data] if uncased else \
               [t[self.TEXT]         for t in self.data]

    def offsets(self):
        """
        Returns a list of [start, end) character offsets of each token.
        """
        return [t[self.SPAN] for t in self.data]
    
    def _get_annotation(self, key: str, index: int):
        if key not in self.annotators:
            return None
        return [t[index] for t in self.data]
    
    def pos(self):
        return self._get_annotation("pos", self.POS)
    def lemmas(self):
        return self._get_annotation("lemma", self.LEMMA)
    def entities(self):
        return self._get_annotation("ner", self.NER)

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """
        Returns a list of ngrams.
        
        Args:
            n          (int): Upper limit of ngram.
            uncased   (bool): Lowercases text or not.
            filter_fn       : User function that filters ngrams.
            as_string (bool): Return the ngram as a string or not.
        """
        words  = self.word(uncased)
        ngrams = []
        
        for s in range(len(words)):
            for e in range(s, min(s + n, len(words))):
                gram = words[s : e+1]
                if not filter_fn or not filter_fn(gram):
                    ngrams.append(gram)
            
        return [" ".join(words[s:e]) for (s, e) in ngrams] if as_strings else ngrams

    def entity_groups(self):
        """
        Group consecutive entity tokens with the same NER tag.
        """
        entities = self.entities()
        if not entities:
            return None
        
        non_ent = self.opts.get("non_ent", "O")
        groups, i = [], 0
        while i < len(entities):
            ner_tag = entities[i]
            if ner_tag != non_ent:
                s = i
                while i < len(entities) and entities[i] == ner_tag:
                    i += 1
                groups.append((self.slice(s, i).untokenize(), ner_tag)) 
            else:
                i += 1
        return groups

class Tokenizer(object):
    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()

class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS    = r"[^\p{Z}\p{C}]"

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(f"({self.ALPHA_NUM})|({self.NON_WS})",
                                     flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE)
        
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning(f"{type(self).__name__} only tokenizes! Skipping annotators: {kwargs.get('annotators')}")
        
        # SimpleTokenizer doesn't support annotators
        self.annotators = set()

    def tokenize(self, text):
        data    = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span     = matches[i].span()
            start_ws = span[0]
            end_ws   = matches[i + 1].span()[0] if i + 1 < len(matches) else span[1]

            # Format data
            data.append((token, text[start_ws:end_ws], span))
            
        return Tokens(data, self.annotators)

# TODO: Thử với SpacyTokenizer
# NOTE: Tốn thời gian ~10 lần so với SimpleTokenizer
class SpacyTokenizer(Tokenizer):
    pass