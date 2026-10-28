# Which language each sample is written in: function words per language, with Chinese and code apart.
import re
import sys
from collections import Counter
from pathlib import Path

W = {'es': 'de la que el en y los se no las por un una con es lo del me te mi su porque pero como más yo tú ella hay',
     'it': 'il che di non per sono è della gli anche come più questo questa ma nel alla perché',
     'fr': 'le les est et des une pas je il qui dans du au ce sur mais avec pour vous nous',
     'en': 'the and of to is in that it for you with was this are be not on'}
W = {k: set(v.split()) for k, v in W.items()}


def lang(s):
    if len(re.findall(r'[一-鿿]', s)) > 20:
        return 'zh'
    if re.search(r'\bdef |\bimport |[{};]\s*\n|==|\breturn\b', s):
        return 'code'
    words = re.findall(r"[a-záéíóúñüàèìòùçâêîôû']+", s.lower())
    c = {k: sum(w in v for w in words) for k, v in W.items()}
    return max(c, key=c.get) if max(c.values()) >= 5 else 'other'


SEP = '\n---------------\n'
for f in sys.argv[1:]:
    ss = [p for p in Path(f).read_text(encoding='utf-8').split(SEP) if p.strip()]
    print(Path(f).name, len(ss), dict(Counter(lang(s) for s in ss).most_common()))
