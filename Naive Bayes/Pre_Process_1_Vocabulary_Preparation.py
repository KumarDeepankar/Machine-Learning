from nltk.tokenize import WhitespaceTokenizer
train_txt = open('traindata')
stop_wrd_txt = open('stoplist')
stop_wrd_set = set()
train_dictionary = set()

for line1 in stop_wrd_txt:
        stop_wrd_set.add(line1.strip())


for line in train_txt:
    token = WhitespaceTokenizer().tokenize(line)
    for wrd in token:
        if wrd not in stop_wrd_set:
            train_dictionary.add(wrd)

final_dic = sorted(train_dictionary)
write_dic = open('Dictionary', 'w')

for wrd in final_dic:
    write_dic.write(wrd+"\n")

write_dic.close()
stop_wrd_txt.close()
train_txt.close()