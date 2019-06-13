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

# Load an spacy model (supported models are "es" and "en") 
nlp = spacy.load('en')
nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')
token = nlp('upset stress')

# for tok in token:
# 	print(tok._.wordnet.synsets())
# 	print(tok._.wordnet.lemmas())
# 	print(tok._.wordnet.wordnet_domains())

# Imagine we want to enrich the following sentence with synonyms
sentence = nlp('I would like to cope with anxiety')

# spaCy WordNet lets you find synonyms by domain of interest
# for example economy
psyc_domains = ['psychological_features','psychology','psychoanalysis']
enriched_sentence = []

# For each token in the sentence
for token in sentence:
    # We get those synsets within the desired domains
    synsets = token._.wordnet.wordnet_synsets_for_domain(psyc_domains)
    if synsets:
        lemmas_for_synset = []
        for s in synsets:
            # If we found a synset in the economy domains
            # we get the variants and add them to the enriched sentence
            lemmas_for_synset.extend(s.lemma_names())
            enriched_sentence.append('({})'.format('|'.join(set(lemmas_for_synset))))
    else:
        enriched_sentence.append(token.text)

# Let's see our enriched sentence
print(' '.join(enriched_sentence))
# >> I (need|want|require) to (draw|withdraw|draw_off|take_out) 5,000 euros

from spacy import displacy
html = displacy.render(doc, style="ent", page=True,
                       options={"ents": ["EVENT"]})

# importlib.import_module('keyword_2')

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

patterns = [
    {"label": "ORG", "pattern": "Apple"},
    {"label": "GPE", "pattern": [{"lower": "san"}, {"lower": "francisco"}]}
]


sentiment_analyzer = SentimentIntensityAnalyzer()

def polarity_scores(doc):
	return sentiment_analyzer.polarity_scores(doc.text)

Doc.set_extension('polarity_scores', getter=polarity_scores)

nlp = spacy.load('en_core_web_lg')

ruler = EntityRuler(nlp, overwrite_ents=True)
ruler.add_patterns(patterns)
nlp.add_pipe(ruler)

wn_pipeline = WordnetPipeline(nlp)
nlp.add_pipe(wn_pipeline, name='wn_synsets')

body = "Terrible. Appointments were too sparse. It seemed like the counsellor forgot me first and I had to reintroduce myself and repeat everything every single session."
body = keyword_2.cleaner(body)
# print(body)

#ipdb.set_trace()
# doc = nlp(body)

# for token in doc:
#     print(token.text, token.has_vector, token.vector_norm, token.is_oov)
# # doc = nlp(u"Apple is opening its first big office in San Francisco.")
# print([(ent.text, ent.label_) for ent in doc.ents])
# print(ruler.labels)

# print(doc._.polarity_scores)
# print(nlp.vocab['banana'].vector)

# for token in doc:
	# print(token.text, "-", token._.synset)
	# print(token.labels)


# nlp = spacy.load('en_core_web_md')  # make sure to use larger model!
tokens1 = nlp(u'Be able to control negative thoughts')
tokens2 = nlp(u'Having someone to talk to')
tokens3 = nlp(u'Increasing confidence ')
cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x,y)

# for token1 in tokens1:
#     for token2 in tokens2:
#         print("Sim: ", token1.text, token2.text, token1.similarity(token2))
        # print("Cosine: ", token1.text, token2.text, cosine_similarity(token1.vector,token2.vector))
print(tokens1.similarity(tokens2))
print(tokens1.similarity(tokens3))
print(tokens3.similarity(tokens2))

anxiety = nlp.vocab['anxiety']
# stress = nlp.vocab['stress']

computed_sims = []

for word in nlp.vocab:
	if not word.has_vector: 
		continue
	similarity = cosine_similarity(anxiety, word.vector)
	computed_sims.append((word, similarity))

computed_sims = sorted(computed_sims, key=lambda item: -item[1])
print([w[0].text for w in computed_sims[:10]])

# nlp = spacy.load('en')
# doc = nlp("This group will introduce participants to a variety of cognitive-behavioural therapy techniques to help manage anxiety and depression.")

# print([(token.text, token.tag_) for token in doc])
# print([(ent) for ent in doc.ents])
# for ent in doc.ents:
# 	print(ent.text, ent.label_)

# iob_tagged = [
# (
# 	token.text,
# 	token.tag_,
# 	"{0}-{1}".format(token.ent_iob_, token.ent_type_) 
# 	if token.ent_iob_ != '0' else token.ent_iob_
# 	) for token in doc
# 	]

# print(iob_tagged)

# for token in doc:
#     print("{0}/{1} <--{2}-- {3}/{4}".format(
#         token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))