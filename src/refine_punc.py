import re


def refine_punc(input_text):
    text = ""
    for i in ' '.join(input_text).replace(' ##', '').split():
        if re.match('\W', i) is not None and re.match('\s', i) is None:
            text += i
        else:
            text += (' '+i)
    for j in ['[', '‘', '{', '<', '(', '“']:
        text = text.replace(j+' ', ' '+j)
    text = ' '.join(['"'+i.strip()+'"' if idx % 2 != 0 else i.strip() for idx, i in enumerate(text.split('"')) ])
    text = ' '.join(["'"+i.strip()+"'" if idx % 2 != 0 else i.strip() for idx, i in enumerate(text.split("'")) ])
    text = text.replace(' [UNK] ', '').replace('[UNK] ', '').replace(' [UNK]', '').replace('[UNK]', '')
    text = text.replace('>', '')
    return text
