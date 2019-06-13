import spacy
import ipdb
import keyword_2
from spacy.tokens import Doc
from spacy.tokens import Token

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn 
# nltk.download('vader_lexicon')

from scipy import spatial
from spacy.pipeline import EntityRuler

from spacy_wordnet.wordnet_annotator import WordnetAnnotator 

def penn_to_wn(tag):
	if tag.startswith('N'):
		return 'n'

	if tag.startswith('V'):
		return 'v'

	if tag.startswith('J'):
		return 'a'

	if tag.startswith('R'):
		return 'r'

	return None

class WordnetPipeline(object):

	def __init__(self, nlp):
		Token.set_extension('synset', default=None)

	def __call__(self,doc):

		for token in doc:
			wn_tag = penn_to_wn(token.tag_)
			ss = wn.synsets(token.text, wn_tag)
			if wn_tag is not None and len(ss) > 0:
				ss = wn.synsets(token.text, wn_tag)
				token._.set('synset', ss)

		return doc


def polarity_scores(doc):
	sentiment_analyzer = SentimentIntensityAnalyzer()
	return sentiment_analyzer.polarity_scores(doc.text)

def phrase_sim(target, phrase):
	return target.similarity(phrase)

def get_top_sims(keyword, doc, show=False):
	# returns 
	cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x,y)
	computed_sims = []

	for word in doc.vocab:
		if not word.has_vector:
			continue

		similarity = cosine_similarity(keyword.vector,word.vector)
		computed_sims.append((word, similarity))

	computed_sims = sorted(computed_sims, key=lambda item: -item[1])

	if show is True:
		print([w[0].text for w in computed_sims[:10]])

	return computed_sims

def enriched_sentence(sentence, domains, show=False):
	enriched_sentence = []

	for token in sentence:
		synsets = token._.wordnet.wordnet_synsets_for_domain(domains)
		if synsets:
			lemmas_for_synsets = []
			for s in synsets:
				lemmas_for_synsets.extend(s.lemma_names())
				enriched_sentence.append('({})'.format('|'.join(set(lemmas_for_synsets))))
		else:
			enriched_sentence.append(token.text)

	if show is True:
		print(' '.join(enriched_sentence))

	return enriched_sentence

##TESTING
nlp = spacy.load('en_core_web_lg')

nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')
nlp.add_pipe(WordnetPipeline(nlp), name='wn_synsets')

body = "I want to cope with anxiety".lower()
sentence = nlp(body)

domains = ['psychology', 'psychological_features']
# enriched_sentence(sentence, domains,show=True)

# ruler = EntityRuler(nlp, overwrite_ents=True)
# ruler.add_patterns(patterns)
# nlp.add_pipe(ruler)

# get_sorted_sims(sentence[3],sentence,show=True)


Doc.set_extension('polarity_scores', getter=polarity_scores)
# print(doc._.polarity_scores)
