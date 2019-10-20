
punc={'.',':','؟','*','"','#','!','،','؛','_','-','+','=','«','»','(',')','{','}','[',']'}
prep_conj={'از','تا','را','كه','که','يا','یا','بر','و','براي','برای','براى','مگر','مثل','مانند','الا','جز','چون','غير','غیر','زير','زیر','زيرا','زیرا','ليکن','لیکن','بهر','اگر','درباره','مقابل','برابر','طوری‌که','طوري‌که','پيش','پیش','پس'}
stop={'ها','اي','مي','خواهد','خواهم','خواهي','خواهيم','خواهيد','خواهند','است','نيست','نیست','هست','بود','شد','باشد','شود','شده','مي‌شود','می‌شود','کرد','كرد','كرده','نكرده','مي‌کند','می‌کند','مي‌کنند','می‌کنند','دارد','ندارد','هر','هم','خود','هيچ','هیچ','هميشه','همیشه','مرا','من','تو','او','ما','شما','ايشان','ایشان','وي','وی','آن','اين','این','آنان','آن‌ها','این‌ها','این‌ها','خاطر','وسيله','وسیله','جهت','چرا','حالا','بسيار','بسیار','برخي','برخی','برخى','بعضي','بعضی','بعضى','شايد','شاید','همين','همین','همان','همچنين','همچنین','همچنان','بايستي','بایستی','مي‌بايست','می‌بایست','باره'}

from features_list_2 import * 
from wsd_shir_1_train import sens_prob, tag_types

import nltk
import math
import codecs

# خواندن اطلاعات از فايل متني ليست تصميم
############################################

fp_train_sorted_decision_list = codecs.open('shir_1_train_sorted_decision_list.txt','r','utf_8')  #* اشاره گر به فايل متني که ليست تصميم در آن نوشته مي شود

decision_list_lines=fp_train_sorted_decision_list.read().split('\n')
decision_list_lines_count=len(decision_list_lines) # تعداد خطوط فايل متني ليست تصميم

fp_train_sorted_decision_list.seek(0)

dl_logl_values=   [0 for i in range(decision_list_lines_count)]
dl_feature_types= ['' for i in range(decision_list_lines_count)]
dl_feature_values=['' for i in range(decision_list_lines_count)]
dl_tag_values=    ['' for i in range(decision_list_lines_count)]


for i in range(decision_list_lines_count-1): # مقدار 1- کسرشده به خاطر آخرين کارکتر نيولاين است که بعد از آن چيزي نوشته نشده
    dl_line_str=fp_train_sorted_decision_list.readline()       
    dl_line_words=dl_line_str.split('\t')       
    dl_logl_values[i]=float(dl_line_words[0])    # از فايل متني ليست تصميم و نگهداري آنها در يک ليست جداگانه براي مقايسه LogL خواندن مقادير
    dl_feature_types[i]=dl_line_words[1]
    dl_feature_values[i]=dl_line_words[2]  # خواندن مقادیر فیچرها از فايل متني ليست تصميم و نگهداري آنها در يک ليست جداگانه براي مقايسه
    dl_tag_values[i]=dl_line_words[3]      # خواندن مقادیر برچسبها از فايل متني ليست تصميم و نگهداري آنها در يک ليست جداگانه براي مقايسه


fp_train_sorted_decision_list.close()

# خواندن اطلاعات از فايل متني تست
#############################################

from nltk.corpus import PlaintextCorpusReader
corpus_root = '/'
corpus_root = 'WSD/shir_4 folds_texts/shir_test/' # یا هر نسخه دیگر پایتون که بر روی دستگاه نصب شده) قرارگيرد)  python34 که شامل زیرفولدرهاي ذکر شده است بايد حتما در دايرکتوري WSD فولدر
peykare = PlaintextCorpusReader(corpus_root, '.*') 

f=peykare.fileids()
l=len(f)

fp=peykare.open(f[0])

peykare_lines=fp.read().split('\n')
peykare_lines_count=len(peykare_lines) # تعداد خطوط پيکره

fp.seek(0)


test_given_lines_tags=['' for i in range(peykare_lines_count)]
for i in range(peykare_lines_count):
    line_str=fp.readline()       
    line_words=line_str.split()       
    test_given_lines_tags[i]=line_words[-1]     # test_given_lines_tags ذخیره‌کردن برچسب خطوط فايل در بردار  

fp.seek(0)

appropriate_sens= ['' for i in range(peykare_lines_count)]

for i in range(peykare_lines_count):
    
    line_str=fp.readline()          # متن (سترينگ) هر خط از فايل
    if i==0:
        line_str=line_str[1:]
    line_words= line_str.split()    # ليست کلمات هر خط فايل
    line_words= line_words[:-1]    # جداکردن برچسب از انتها و کاراکتر یونی‌کد از ابتدای لیست کلمات هر خط فايل 
    line_words_set=set(line_words)  # مجموعه کلمات هر خط فايل
    line_words1= list(line_words_set.difference(punc).difference(stop).difference(prep_conj)) # لیست  مجموعه کلمات هر خط فايل پس از حذف پانکچوئيشنها، ستاپ وردز، حروف ربط و حروف اضافه

    main_line_words=[]
    for w in line_words :
        if w in line_words1:
            main_line_words.append(w)  #  ليست کلمات خط خوانده‌شده بدون پانکچوئيشنها، ستاپ وردز و حروف ربط یا اضافه


    #  پيش بيني برچسب خط با استفاده از فيچرهاي کالوکيشنال مانند
    ## کلمات قبل و بعد(ماينس و پلاس) تارگت-ورد و باهم آيي سه تايي ( ترايگرم) کلمات
    ####################################################################################
    
    for w in line_words:
        if 'شير'==w  or ('شير' in w and (w!='شيريني' and w!='شيرين' and w!= 'شمشير')): # درنظرگرفتن خود تارگت-ورد و تصريفات و ترکيبات مرتبط آن و رد مقادیر غیرمرتبط
            w_index= line_words.index(w) # به دست آوردن انديس تارگت-ورد در ليست کلمات خط بدون حذف پانکچوئيشنها، ستاپ -وردز و حروف ربط و اضافه

    for j in range(len(dl_feature_types)):
        if appropriate_sens[i]=='':
            if dl_feature_types[j]=='shir_trif':
                if dl_feature_values[j] in line_str:
                    appropriate_sens[i]= dl_tag_values[j]

    if appropriate_sens[i]=='':
        for j in range(len(dl_feature_types)):
            if appropriate_sens[i]=='':
                if dl_feature_types[j]=='shir_plus1f' :
                    if (len(line_words[w_index:])> 1):
                        if (line_words[w_index+1] == dl_feature_values[j]) or (dl_feature_values[j] in line_words[w_index+1]):
                            appropriate_sens[i]= dl_tag_values[j]

    if appropriate_sens[i]=='':
        for j in range(len(dl_feature_types)):
            if appropriate_sens[i]=='':
                if dl_feature_types[j]=='shir_minus1f' :
                    if (w_index!= 0):
                        if (line_words[w_index-1] == dl_feature_values[j]) or (dl_feature_values[j] in line_words[w_index-1]):
                            appropriate_sens[i]= dl_tag_values[j]

    
    # context window به دست آوردن کلمات اطراف تارگت-ورد با توجه به سايزهاي مختلف  

    for w in main_line_words:
        if 'شير'==w  or ('شير' in w and (w!='شيريني' and w!='شيرين' and w!= 'شمشير')): # درنظرگرفتن خود تارگت-ورد و تصريفات و ترکيبات مرتبط آن و رد مقادیر غیرمرتبط
            w_index= main_line_words.index(w) # به دست آوردن انديس تارگت-ورد در ليست کلمات خط بدون حذف پانکچوئيشنها، ستاپ -وردز و حروف ربط و اضافه


            # در نظر گرفتن پنجره ±5 تايي براي مقايسه کلمات اطراف تارگت-ورد با ليست(بردار) فیچرها

            if w_index == 0 :                  #تارگت-ورد در ابتداي (مکان صفرم) ليست کلمات خط باشد
                if len(main_line_words)<= 6:   # طول ليست بعد از تارگت-ورد کمتر از 5+1 باشد
                    comapre_list5= main_line_words[1:]
                    compare_list10=[]
                    compare_list10m=[]
                                  
                else:
                    compare_list5= main_line_words[1:6]
                    if len(main_line_words)<= 11:
                        comapre_list10= main_line_words[6:]
                        comapre_list10m= []
                    else:
                        comapre_list10= main_line_words[6:11]
                        comapre_list10m= main_line_words[11:]
                        

            elif w_index==1 or w_index==2 or w_index==3 or w_index==4 or w_index==5:  # تارگت-ورد در مکان اول تا پنجم در ليست باشد
                if len(main_line_words[w_index:]) <= 6:         # طول ليست بعد از تارگت-ورد کمتر از 5+1 باشد
                    if len(main_line_words[w_index:])==1 :      # تارگت-ورد، کلمه آخر ليست کلمات خط نيز باشد
                        comapre_list5= main_line_words[:w_index]
                    else:
                        comapre_list5= main_line_words[:w_index] + main_line_words[w_index+1:]

                    compare_list10=[]
                    compare_list10m=[]

                else:
                    comapre_list5= main_line_words[:w_index] + main_line_words[w_index+1:w_index+6]

                    if len(main_line_words[w_index:])<= 11:          # طول ليست بعد از تارگت-ورد کمتر از 10+1 باشد
                        comapre_list10= main_line_words[w_index+6:]
                        comapre_list10m=[]
                    else:
                        comapre_list10= main_line_words[w_index+6:w_index+11] # ليست کلمات جايگاههاي ششم تا دهم بعد از تارگت-ورد
                        comapre_list10m= main_line_words[w_index+11:]         # ليست کلمات جايگاه يازدهم بعد از تارگت-ورد تا انتهاي  خط 


            elif w_index > 5 :
                if len(main_line_words[w_index:]) <= 6:    # تارگت-ورد در مکان ششم يا بيشتر در ليست قرارگرفته و بيش از 5 کلمه قبل از آن در ليست وجودداشته‌باشد
                    if (len(main_line_words[w_index:])==1) :
                        comapre_list5= main_line_words[w_index-5:w_index]
                    else:
                        comapre_list5= main_line_words[w_index-5:w_index] + main_line_words[w_index+1:]

                    if len(main_line_words[:w_index+1])<= 11:
                        comapre_list10= main_line_words[:w_index-5]   # (ليست کلمات ابتدای خط تا جايگاه ششم قبل از تارگت-ورد (کمتر از 5 عدد
                        comapre_list10m=[]
                    else:
                        comapre_list10= main_line_words[w_index-10:w_index-5] # ليست کلمات جايگاههاي ششم تا دهم قبل از تارگت-ورد
                        comapre_list10m= main_line_words[:w_index-10]         # ليست کلمات ابتدای خط تا جايگاه یازدهم قبل از تارگت-ورد  

                        
                else:
                    comapre_list5= main_line_words[w_index-5:w_index] + main_line_words[w_index+1:w_index+6]

                    if len(main_line_words[:w_index+1])<= 11 and len(main_line_words[w_index:]) <= 11 :
                        comapre_list10= main_line_words[:w_index-5] + main_line_words[w_index+6:]
                        comapre_list10m=[]

                    elif len(main_line_words[:w_index+1])<= 11 and len(main_line_words[w_index:]) > 11 :
                        comapre_list10= main_line_words[:w_index-5] + main_line_words[w_index+6:w_index+11]
                        comapre_list10m= main_line_words[w_index+11:]

                    elif len(main_line_words[:w_index+1])> 11 and len(main_line_words[w_index:]) <= 11 :
                        comapre_list10= main_line_words[w_index-10:w_index-5] + main_line_words[w_index+6:]
                        comapre_list10m= main_line_words[:w_index-10]

                    elif len(main_line_words[:w_index+1])> 11 and len(main_line_words[w_index:]) > 11 :
                        comapre_list10= main_line_words[w_index-10:w_index-5] + main_line_words[w_index+6:w_index+11]
                        comapre_list10m= main_line_words[:w_index-10] + main_line_words[w_index+11:]


    ## در ليست تصميم LogL  و به ترتيب مقادير bag of words   پيش بيني برچسب با استفاده از رخداد تکي يا بايگرم  
    #########################################################################################################

    if appropriate_sens[i]=='':
        for j in range(len(dl_feature_values)):
            for k in range(len(comapre_list5)):
                if appropriate_sens[i]=='':
                    if (dl_feature_values[j]==comapre_list5[k]) or (dl_feature_values[j] in comapre_list5[k]):
                        appropriate_sens[i]= dl_tag_values[j]

    if appropriate_sens[i]=='':
        if comapre_list10 !=[]:
            for j in range(len(dl_feature_values)):
                for k in range(len(comapre_list10)):
                    if appropriate_sens[i]=='':
                        if (dl_feature_values[j]==comapre_list10[k]) or (dl_feature_values[j] in comapre_list10[k]):
                            appropriate_sens[i]= dl_tag_values[j]

    if appropriate_sens[i]=='':
        if comapre_list10m !=[]:
            for j in range(len(dl_feature_values)):
                for k in range(len(comapre_list10m)):
                    if appropriate_sens[i]=='':
                        if (dl_feature_values[j]==comapre_list10m[k]) or (dl_feature_values[j] in comapre_list10m[k]):
                            appropriate_sens[i]= dl_tag_values[j]

    
    if appropriate_sens[i]=='':
        for j in range(len(dl_feature_types)):
            if appropriate_sens[i]=='':
                if dl_feature_types[j]=='shir_bif' and (dl_feature_values[j] in line_str):
                    appropriate_sens[i]= dl_tag_values[j]

    if appropriate_sens[i]=='':
        appropriate_sens[i]= tag_types[sens_prob.index(max(sens_prob))] # اگر جمله/پاراگرافي در پيکره تست وجودداشته باشد که هيچ يک از کلمات آن در ليست تصميم موجود نباشد
                                                                        ## با برچسبي که بيشترين فراواني را در پيکره آموزش دارد، برچسب دهي مي شود
        

fp.close()

true_positive=0
for i in range(len(appropriate_sens)):
    if (appropriate_sens[i]== test_given_lines_tags[i]):
        true_positive+=1     # تعداد برچسب هاي صحيح

#* Cross_Validation ام در -k به دست آوردن دقت برچسب دهي تست

Precision= (true_positive/ len(appropriate_sens))*100

print('shir_test_1 Precision =', Precision)
print()
