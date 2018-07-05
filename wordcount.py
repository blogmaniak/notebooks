import re
import string
from operator import itemgetter
from collections import OrderedDict

readfiles = 5

for x in range(readfiles):
  frequency = {}
  document_text = open('../data/tweets_'+str(x)+'.txt', 'r')
  text_string = document_text.read().lower()
  match_pattern = re.findall(r'\b[a-z]{3,15}\b', text_string)

for word in match_pattern:
  count = frequency.get(word,0)
  frequency[word] = count + 1

frequency = OrderedDict(sorted(frequency.items(), key=lambda t: t[1], reverse=True))

frequency_list = frequency.keys()

print"*** tweets_"+str(x)+"************"

for words in frequency_list:
  print words, frequency[words]

print"*********************************"

