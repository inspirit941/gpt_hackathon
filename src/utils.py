import re


def convert_text(text):
   '''
   convert text for sentence split
   split token: '.. '
   '''
   out = re.sub('[\n]+', '',
                re.sub('[?]', '?.. ',
                       re.sub('다\.', '다... ',
                              re.sub('\."', '"',
                                     re.sub('[!]', '!.. ',
                                            re.sub('[.?]+', '.',
                                                   re.sub('[ ]+', ' ',
                                                          re.sub('[.]+', '.', text))))))))
   return out


def rm_sp(x):
    while 1:
        try:
            x.remove('')
        except:
            try:
                x.remove(' ')
            except:
                break
    return x
