
punc={'.',':','؟','*','"','#','!','،','؛','_','-','+','=','«','»','(',')','{','}','[',']'}
prep_conj={'از','تا','را','كه','که','يا','یا','بر','و','براي','برای','براى','مگر','مثل','مانند','الا','جز','چون','غير','غیر','زير','زیر','زيرا','زیرا','ليکن','لیکن','بهر','اگر','درباره','مقابل','برابر','طوری‌که','طوري‌که','پيش','پیش','پس'}
stop={'ها','اي','مي','خواهد','خواهم','خواهي','خواهيم','خواهيد','خواهند','است','نيست','نیست','هست','بود','شد','باشد','شود','شده','مي‌شود','می‌شود','کرد','كرد','كرده','نكرده','مي‌کند','می‌کند','مي‌کنند','می‌کنند','دارد','ندارد','هر','هم','خود','هيچ','هیچ','هميشه','همیشه','مرا','من','تو','او','ما','شما','ايشان','ایشان','وي','وی','آن','اين','این','آنان','آن‌ها','این‌ها','این‌ها','خاطر','وسيله','وسیله','جهت','چرا','حالا','بسيار','بسیار','برخي','برخی','برخى','بعضي','بعضی','بعضى','شايد','شاید','همين','همین','همان','همچنين','همچنین','همچنان','بايستي','بایستی','مي‌بايست','می‌بایست','باره'}

from features_list_2 import * 

import nltk
import math
import codecs

from nltk.corpus import PlaintextCorpusReader
corpus_root = '/'
corpus_root = 'WSD/krm_4 folds/krm_train/' # قرارگيرد  python34 که شامل زیرفولدرهاي ذکر شده است بايد حتما در دايرکتوري WSD فولدر
peykare = PlaintextCorpusReader(corpus_root, '.*') 

f=peykare.fileids()
l=len(f)


fp=peykare.open(f[3])
peykare_lines=fp.read().split('\n')
peykare_lines_count=len(peykare_lines) # تعداد خطوط پيکره

fp.seek(0)

peykare_words=fp.read().split()    # کل کلمات پيکره


fp.seek(0)

lines_tags=['' for i in range(peykare_lines_count)]
for i in range(peykare_lines_count):
    line_str=fp.readline()       
    line_words=line_str.split()       
    lines_tags[i]=line_words[-1]     # lines_tags ذخیره‌کردن برچسب خطوط فايل در بردار  

tag_types=list(set(lines_tags))      # جملات (خطوط) فايل (sens-types) ليست نوع-برچسبهاي

sens_count=[0 for j in range(len(tag_types))]
sens_prob= [0 for j in range(len(tag_types))]

for j in range(len(tag_types)):
    sens_count[j]= lines_tags.count(tag_types[j])  #* Count(s[j]) : تعداد هر برچسب در پيکره آموزش
    sens_prob[j]= sens_count[j]/len(lines_tags)    #* Probability (s[j]) : احتمال هر برچسب در پيکره آموزش

#@@@ 

# لیست کل کلمات متن پس از حذف برچسبها، پانکچوئيشنها، ستاپ وردز، حروف ربط و حروف اضافه
main_peykare_words= list(set(peykare_words).difference(set(lines_tags)).difference(punc).difference(stop).difference(prep_conj)) 

  

fp.seek(0)

#  با تارگت-ورد callocational تعريف ماتريسهاي شمارش فيچرهاي 
krm_plus1f_sens_count_matrix= [[0 for i in range(len(tag_types))] for j in range(len(krm_calloc_i_plus1_features))]
krm_minus1f_sens_count_matrix=[[0 for i in range(len(tag_types))] for j in range(len(krm_calloc_i_minus1_features))]

#  bag of words تعريف ماتريسهاي شمارش فيچرهاي 
krm_bowf5_sens_count_matrix= [[0 for i in range(len(tag_types))] for j in range(len(krm_bagofwords_features5))]
krm_bowf10_sens_count_matrix= [[0 for i in range(len(tag_types))] for j in range(len(krm_bagofwords_features10))]
krm_bowf10m_sens_count_matrix= [[0 for i in range(len(tag_types))] for j in range(len(krm_bagofwords_features10more))]

# (با يکديگر (نه نسبت به تارگت-ورد callocatioal تعريف ماتريس شمارش فيچرهاي  
krm_bi_sens_count_mtrix= [[0 for i in range(len(tag_types))] for j in range(len(krm_bigrams))]
krm_tri_sens_count_mtrix=[[0 for i in range(len(tag_types))] for j in range(len(krm_trigrams))]

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

    # تعیین شماره‌ای (اندیس) معادل براي برچسب خط (خوانده‌شده از فایل)، براي استفاده در ماتريسهاي شمارش و احتمال
    for tt in tag_types:
        if lines_tags[i]==tt:
            sens_index=tag_types.index(tt)

    # به دست آوردن شمارش فيچرهاي کالوکيشنال مانند کلمات قبل و بعد(ماينس و پلاس) تارگت-ورد
    ##                         و باهم آيي هاي دو تايي و سه تايي (بايگرم و ترايگرم) کلمات
    ####################################################################################
    
    for w in line_words:
        if 'کرم'==w  or ('کرم' in w and (w!='شاکرم' and w!='چاکرم' and w!='کرمان')): # درنظرگرفتن خود تارگت-ورد و تصريفات و ترکيبات مرتبط آن و رد مقادیر غیرمرتبط

            w_index= line_words.index(w) # به دست آوردن انديس تارگت-ورد در ليست کلمات خط بدون حذف پانکچوئيشنها، ستاپ -وردز و حروف ربط و اضافه

            if w_index == 0 :                                             #تارگت-ورد در ابتداي (مکان صفرم) ليست کلمات خط باشد
                for f in krm_calloc_i_plus1_features:
                    if f==line_words[w_index+1] or f in line_words[w_index+1]:
                        #@@ بايد فيچرهاي متعلق به اين قسمت اصلاح شوند تا کانت کلا صفر نباشدfor l in lis:
                        krm_plus1f_sens_count_matrix[krm_calloc_i_plus1_features.index(f)][sens_index]+=1 # ماتريس شمارشهاي اوليه فيچرهاي يکي بعد از تارگت-وردها

            elif len(line_words[w_index:])==1 :                            # تارگت-ورد کلمه آخر ليست کلمات خط نيز باشد
                for f in krm_calloc_i_minus1_features:
                    if f==line_words[w_index-1] or f in line_words[w_index-1]:
                        #@@ بايد فيچرهاي متعلق به اين قسمت اصلاح شوند تا کانت کلا صفر نباشد
                        krm_minus1f_sens_count_matrix[krm_calloc_i_minus1_features.index(f)][sens_index]+=1 # ماتريس شمارشهاي اوليه فيچرهاي يکي قبل از تارگت-وردها
        
            elif w_index > 0 and len(line_words[w_index:])!=1 :
                for f in krm_calloc_i_plus1_features:
                    if f==line_words[w_index+1] or f in line_words[w_index+1]:
                        #@@ بايد فيچرهاي متعلق به اين قسمت اصلاح شوند تا کانت کلا صفر نباشد
                        krm_plus1f_sens_count_matrix[krm_calloc_i_plus1_features.index(f)][sens_index]+=1 # ماتريس شمارشهاي اوليه فيچرهاي يکي بعد از تارگت-وردها

                for f in krm_calloc_i_minus1_features:
                    if f==line_words[w_index-1] or f in line_words[w_index-1]:
                        #@@ بايد فيچرهاي متعلق به اين قسمت اصلاح شوند تا کانت کلا صفر نباشد
                        krm_minus1f_sens_count_matrix[krm_calloc_i_minus1_features.index(f)][sens_index]+=1 # ماتريس شمارشهاي اوليه فيچرهاي يکي قبل از تارگت-وردها

    #@@ بايد فيچرهاي متعلق به اين قسمت اصلاح شوند تا کانت کلا صفر نباشد
    for b in krm_bigrams:
        if b in line_str:
            krm_bi_sens_count_mtrix[krm_bigrams.index(b)][sens_index]+=1 # ماتريس اولیه شمارش باهم آيي دو‌تايي فيچرها که بايد مقادیری از آن که به ازای تمامی برچسبها صفر هستند، بعدا اصلاح شوند

    for t in krm_trigrams:
        if t in line_str:
            krm_tri_sens_count_mtrix[krm_trigrams.index(t)][sens_index]+=1 # ماتريس اولیه شمارش باهم آيي سه‌تايي فيچرها که بايد بعدا اصلاح شود
    

    
    ## bag of words به دست آوردن شمارش فيچرهاي 
    ####################################################################################

    # context window به دست آوردن کلمات اطراف تارگت-ورد با توجه به سايزهاي مختلف  

    for w in main_line_words:
        if 'کرم'==w  or ('کرم' in w and (w!='شاکرم' and w!='چاکرم' and w!='کرمان')) : # درنظرگرفتن خود تارگت-ورد و تصريفات و ترکيبات مرتبط آن و رد مقادیر غیرمرتبط
            
            w_index= main_line_words.index(w) # به دست آوردن انديس تارگت-ورد در ليست کلمات خط بدون حذف پانکچوئيشنها، ستاپ -وردز و حروف ربط و اضافه


            # در نظر گرفتن پنجره ±5 تايي براي مقايسه کلمات اطراف تارگت-ورد با ليست(بردار) فیچرها

            if w_index == 0 :                  #تارگت-ورد در ابتداي (مکان صفرم) ليست کلمات خط باشد
                if len(main_line_words)<= 6:   # طول ليست بعد از تارگت-ورد کمتر از 5+1 باشد
                    compare_list5= main_line_words[1:]
                    compare_list10=[]
                    compare_list10m=[]
                                  
                else:
                    compare_list5= main_line_words[1:6]
                    if len(main_line_words)<= 11:
                        compare_list10= main_line_words[6:]
                        compare_list10m= []
                    else:
                        compare_list10= main_line_words[6:11]
                        compare_list10m= main_line_words[11:]
                        

            elif w_index==1 or w_index==2 or w_index==3 or w_index==4 or w_index==5:  # تارگت-ورد در مکان اول تا پنجم در ليست باشد
                if len(main_line_words[w_index:]) <= 6:         # طول ليست بعد از تارگت-ورد کمتر از 5+1 باشد
                    if len(main_line_words[w_index:])==1 :      # تارگت-ورد، کلمه آخر ليست کلمات خط نيز باشد
                        compare_list5= main_line_words[:w_index]
                    else:
                        compare_list5= main_line_words[:w_index] + main_line_words[w_index+1:]

                    compare_list10=[]
                    compare_list10m=[]

                else:
                    compare_list5= main_line_words[:w_index] + main_line_words[w_index+1:w_index+6]

                    if len(main_line_words[w_index:])<= 11:          # طول ليست بعد از تارگت-ورد کمتر از 10+1 باشد
                        compare_list10= main_line_words[w_index+6:]
                        compare_list10m=[]
                    else:
                        compare_list10= main_line_words[w_index+6:w_index+11] # ليست کلمات جايگاههاي ششم تا دهم بعد از تارگت-ورد
                        compare_list10m= main_line_words[w_index+11:]         # ليست کلمات جايگاه يازدهم بعد از تارگت-ورد تا انتهاي  خط 


            elif w_index > 5 :
                if len(main_line_words[w_index:]) <= 6:    # تارگت-ورد در مکان ششم يا بيشتر در ليست قرارگرفته و بيش از 5 کلمه قبل از آن در ليست وجودداشته‌باشد
                    if (len(main_line_words[w_index:])==1) :
                        compare_list5= main_line_words[w_index-5:w_index]
                    else:
                        compare_list5= main_line_words[w_index-5:w_index] + main_line_words[w_index+1:]

                    if len(main_line_words[:w_index+1])<= 11:
                        compare_list10= main_line_words[:w_index-5]   # (ليست کلمات ابتدای خط تا جايگاه ششم قبل از تارگت-ورد (کمتر از 5 عدد
                        compare_list10m=[]
                    else:
                        compare_list10= main_line_words[w_index-10:w_index-5] # ليست کلمات جايگاههاي ششم تا دهم قبل از تارگت-ورد
                        compare_list10m= main_line_words[:w_index-10]         # ليست کلمات ابتدای خط تا جايگاه یازدهم قبل از تارگت-ورد  

                        
                else:
                    compare_list5= main_line_words[w_index-5:w_index] + main_line_words[w_index+1:w_index+6]

                    if len(main_line_words[:w_index+1])<= 11 and len(main_line_words[w_index:]) <= 11 :
                        compare_list10= main_line_words[:w_index-5] + main_line_words[w_index+6:]
                        compare_list10m=[]

                    elif len(main_line_words[:w_index+1])<= 11 and len(main_line_words[w_index:]) > 11 :
                        compare_list10= main_line_words[:w_index-5] + main_line_words[w_index+6:w_index+11]
                        compare_list10m= main_line_words[w_index+11:]

                    elif len(main_line_words[:w_index+1])> 11 and len(main_line_words[w_index:]) <= 11 :
                        compare_list10= main_line_words[w_index-10:w_index-5] + main_line_words[w_index+6:]
                        compare_list10m= main_line_words[:w_index-10]

                    elif len(main_line_words[:w_index+1])> 11 and len(main_line_words[w_index:]) > 11 :
                        compare_list10= main_line_words[w_index-10:w_index-5] + main_line_words[w_index+6:w_index+11]
                        compare_list10m= main_line_words[:w_index-10] + main_line_words[w_index+11:]


    #@ بعد از اتمام حلقه بايد ليستهاي زير و مشابه هايشان و کليه ماتريسهاي شمارش اصلاح شوند و مقادير صفر آنها حذف گردد
                        
    if len(compare_list5) >= 1:     # شرط هميشه درست براي پنجره 5تايي اطراف تارگت-ورد
        for cw in compare_list5:
            for bowf5 in krm_bagofwords_features5:
                if bowf5==cw or bowf5 in cw:
                    bowf5_index=krm_bagofwords_features5.index(bowf5)
                    krm_bowf5_sens_count_matrix[bowf5_index][sens_index]+=1 #

    if compare_list10 !=[]:   # شرط خالي نبودن ليست کلمات جايگاههاي ششم تا دهم اطراف تارگت-ورد براي مقايسه
        for cw in compare_list10:
            for bowf10 in krm_bagofwords_features10:
                if bowf10==cw or bowf10 in cw:
                    bowf10_index=krm_bagofwords_features10.index(bowf10)
                    krm_bowf10_sens_count_matrix[bowf10_index][sens_index]+=1
                    
    if compare_list10m !=[]:  # شرط خالي نبودن ليست کلمات جايگاه دهم  به بعداطراف تارگت-ورد براي مقايسه
        for cw in compare_list10m:
            for bowf10m in krm_bagofwords_features10more:
                if bowf10m==cw or bowf10m in cw:
                    bowf10m_index=krm_bagofwords_features10more.index(bowf10m)
                    krm_bowf10m_sens_count_matrix[bowf10m_index][sens_index]+=1


fp.close()


# اصلاح ليست فيچرهاي متعلق به اين قسمت از متن که براي کراس-وليديشن انتخاب شده و اصلاح ماتريسهاي شمارش جوينت فيچر و برچسب
######################################################################################################################

#--------------------------------------------------------------------ليستهاي اصلاح شده فيچرها
krm_bowf5=  []
krm_bowf10= []
krm_bowf10m=[]

krm_plus1f= []
krm_minus1f=[]

krm_bif= []
krm_trif=[]
#--------------------------------------------------------------------ماتريسهاي شمارش اصلاح شده
re_krm_bowf5_sens_count_matrix=[]
re_krm_bowf10_sens_count_matrix= []
re_krm_bowf10m_sens_count_matrix= []

re_krm_plus1f_sens_count_matrix= []
re_krm_minus1f_sens_count_matrix=[]

re_krm_bi_sens_count_mtrix= []
re_krm_tri_sens_count_mtrix=[]
#--------------------------------------------------------------------
zero_count=0
for i in range(len(krm_bowf5_sens_count_matrix)):
    for j in range(len(krm_bowf5_sens_count_matrix[i])):
        if (krm_bowf5_sens_count_matrix[i][j]==0):
            zero_count+=1
    if (zero_count!=len(krm_bowf5_sens_count_matrix[i])):
        krm_bowf5.append(krm_bagofwords_features5[i])
        re_krm_bowf5_sens_count_matrix.append(krm_bowf5_sens_count_matrix[i]) # (پرکردن ماتريس شمارش جديد (اصلاح ماتريس شمارش اوليه
    zero_count=0    

        
zero_count=0
for i in range(len(krm_bowf10_sens_count_matrix)):
    for j in range(len(krm_bowf10_sens_count_matrix[i])):
        if (krm_bowf10_sens_count_matrix[i][j]==0):
            zero_count+=1
    if (zero_count!=len(krm_bowf10_sens_count_matrix[i])):
        krm_bowf10.append(krm_bagofwords_features10[i])
        re_krm_bowf10_sens_count_matrix.append(krm_bowf10_sens_count_matrix[i]) # (پرکردن ماتريس شمارش جديد (اصلاح ماتريس شمارش اوليه
    zero_count=0    

            
zero_count=0
for i in range(len(krm_bowf10m_sens_count_matrix)):
    for j in range(len(krm_bowf10m_sens_count_matrix[i])):
        if (krm_bowf10_sens_count_matrix[i][j]==0):
            zero_count+=1
    if (zero_count!=len(krm_bowf10m_sens_count_matrix[i])):
        krm_bowf10m.append(krm_bagofwords_features10more[i])
        re_krm_bowf10m_sens_count_matrix.append(krm_bowf10m_sens_count_matrix[i]) # (پرکردن ماتريس شمارش جديد (اصلاح ماتريس شمارش اوليه
    zero_count=0    


zero_count=0
for i in range(len(krm_plus1f_sens_count_matrix)):
    for j in range(len(krm_plus1f_sens_count_matrix[i])):
        if (krm_plus1f_sens_count_matrix[i][j]==0):
            zero_count+=1
    if (zero_count!=len(krm_plus1f_sens_count_matrix[i])):
        krm_plus1f.append(krm_calloc_i_plus1_features[i])
        re_krm_plus1f_sens_count_matrix.append(krm_plus1f_sens_count_matrix[i]) # (پرکردن ماتريس شمارش جديد (اصلاح ماتريس شمارش اوليه
    zero_count=0    


zero_count=0
for i in range(len(krm_minus1f_sens_count_matrix)):
    for j in range(len(krm_minus1f_sens_count_matrix[i])):
        if (krm_minus1f_sens_count_matrix[i][j]==0):
            zero_count+=1
    if (zero_count!=len(krm_minus1f_sens_count_matrix[i])):
        krm_minus1f.append(krm_calloc_i_minus1_features[i])
        re_krm_minus1f_sens_count_matrix.append(krm_minus1f_sens_count_matrix[i]) # (پرکردن ماتريس شمارش جديد (اصلاح ماتريس شمارش اوليه
    zero_count=0    


zero_count=0
for i in range(len(krm_bi_sens_count_mtrix)):
    for j in range(len(krm_bi_sens_count_mtrix[i])):
        if (krm_bi_sens_count_mtrix[i][j]==0):
            zero_count+=1
    if (zero_count!=len(krm_bi_sens_count_mtrix[i])):
        krm_bif.append(krm_bigrams[i])
        re_krm_bi_sens_count_mtrix.append(krm_bi_sens_count_mtrix[i]) # (پرکردن ماتريس شمارش جديد (اصلاح ماتريس شمارش اوليه
    zero_count=0    

    
zero_count=0
for i in range(len(krm_tri_sens_count_mtrix)):
    for j in range(len(krm_tri_sens_count_mtrix[i])):
        if (krm_tri_sens_count_mtrix[i][j]==0):
            zero_count+=1
    if (zero_count!=len(krm_tri_sens_count_mtrix[i])):
        krm_trif.append(krm_trigrams[i])
        re_krm_tri_sens_count_mtrix.append(krm_tri_sens_count_mtrix[i]) # (پرکردن ماتريس شمارش جديد (اصلاح ماتريس شمارش اوليه
    zero_count=0    


#** براي ليست تصميم log-likelihood ratio محاسبه
################################################################################################
    
decision_list=[]

#-------------------------------------------------------------- re_krm_bowf5_sens_count_matrix	

z=0
for l in re_krm_bowf5_sens_count_matrix:
    for j in range(len(l)):
        if l[j] == 0:
            z+=1
    if z>0:
        for j in range(len(l)):
            l[j]+=1
    z=0

for k in range(len(re_krm_bowf5_sens_count_matrix)):
    
    i=0
    log_likelihood_ratio= [0 for i in range(len(tag_types)*len(tag_types))]

    for m in range(len(re_krm_bowf5_sens_count_matrix[k])):
        for n in range(len(re_krm_bowf5_sens_count_matrix[k])):
            log_likelihood_ratio[i]= math.log10(re_krm_bowf5_sens_count_matrix[k][m]/re_krm_bowf5_sens_count_matrix[k][n])
            i+=1

    #log_likelihood_ratio=[]
    
    #for logl in temp_log_likelihood_ratio:
    #    if (logl > 0): # لگاريتمهاي صفر که نشانه تساوي صورت و مخرج و لگاريتمهاي منفي که نشانه کوچکتر بودن صورت از مخرج هستند، حذف ميشوند
    #        log_likelihood_ratio.append(logl)
            
    max_ratio_index=log_likelihood_ratio.index(max(log_likelihood_ratio))

    if (max_ratio_index%(len(tag_types)-1)==0):
        prefered_sens =tag_types[(max_ratio_index//len(tag_types))-1] # دومین 1-  به خاطر آنکه انديس ليست از صفر شروع مي شود کسر شده است 
    else:
        prefered_sens =tag_types[max_ratio_index//len(tag_types)]

    decision_list.append([max(log_likelihood_ratio)*10000000//10/1000000,'krm_bowf5',krm_bowf5[k],prefered_sens])


#-------------------------------------------------------------------re_krm_bowf10_sens_count_matrix

z=0
for l in re_krm_bowf10_sens_count_matrix:
    for j in range(len(l)):
        if l[j] == 0:
            z+=1
    if z>0:
        for j in range(len(l)):
            l[j]+=1
    z=0

for k in range(len(re_krm_bowf10_sens_count_matrix)):
    
    i=0
    log_likelihood_ratio= [0 for i in range(len(tag_types)*len(tag_types))]

    for m in range(len(re_krm_bowf10_sens_count_matrix[k])):
        for n in range(len(re_krm_bowf10_sens_count_matrix[k])):
            log_likelihood_ratio[i]= math.log10(re_krm_bowf10_sens_count_matrix[k][m]/re_krm_bowf10_sens_count_matrix[k][n])
            i+=1
            
    max_ratio_index=log_likelihood_ratio.index(max(log_likelihood_ratio))

    if (max_ratio_index%(len(tag_types)-1)==0):
        prefered_sens =tag_types[(max_ratio_index//len(tag_types))-1] # دومین 1-  به خاطر آنکه انديس ليست از صفر شروع مي شود کسر شده است 
    else:
        prefered_sens =tag_types[max_ratio_index//len(tag_types)]

    decision_list.append([max(log_likelihood_ratio)*10000000//10/1000000,'krm_bowf10',krm_bowf10[k],prefered_sens])


#-------------------------------------------------------------------re_krm_bowf10m_sens_count_matrix

z=0
for l in re_krm_bowf10m_sens_count_matrix:
    for j in range(len(l)):
        if l[j] == 0:
            z+=1
    if z>0:
        for j in range(len(l)):
            l[j]+=1
    z=0

for k in range(len(re_krm_bowf10m_sens_count_matrix)):
    
    i=0
    log_likelihood_ratio= [0 for i in range(len(tag_types)*len(tag_types))]

    for m in range(len(re_krm_bowf10m_sens_count_matrix[k])):
        for n in range(len(re_krm_bowf10m_sens_count_matrix[k])):
            log_likelihood_ratio[i]= math.log10(re_krm_bowf10m_sens_count_matrix[k][m]/re_krm_bowf10m_sens_count_matrix[k][n])
            i+=1

    max_ratio_index=log_likelihood_ratio.index(max(log_likelihood_ratio))

    if (max_ratio_index%(len(tag_types)-1)==0):
        prefered_sens =tag_types[(max_ratio_index//len(tag_types))-1] # دومین 1-  به خاطر آنکه انديس ليست از صفر شروع مي شود کسر شده است 
    else:
        prefered_sens =tag_types[max_ratio_index//len(tag_types)]

    decision_list.append([max(log_likelihood_ratio)*10000000//10/1000000,'krm_bowf10m',krm_bowf10m[k],prefered_sens])


#-------------------------------------------------------------------re_krm_plus1f_sens_count_matrix

z=0
for l in re_krm_plus1f_sens_count_matrix:
    for j in range(len(l)):
        if l[j] == 0:
            z+=1
    if z>0:
        for j in range(len(l)):
            l[j]+=1
    z=0

for k in range(len(re_krm_plus1f_sens_count_matrix)):
    
    i=0
    log_likelihood_ratio= [0 for i in range(len(tag_types)*len(tag_types))]

    for m in range(len(re_krm_plus1f_sens_count_matrix[k])):
        for n in range(len(re_krm_plus1f_sens_count_matrix[k])):
            log_likelihood_ratio[i]= math.log10(re_krm_plus1f_sens_count_matrix[k][m]/re_krm_plus1f_sens_count_matrix[k][n])
            i+=1

    max_ratio_index=log_likelihood_ratio.index(max(log_likelihood_ratio))

    if (max_ratio_index%(len(tag_types)-1)==0):
        prefered_sens =tag_types[(max_ratio_index//len(tag_types))-1] # دومین 1-  به خاطر آنکه انديس ليست از صفر شروع مي شود کسر شده است 
    else:
        prefered_sens =tag_types[max_ratio_index//len(tag_types)]

    decision_list.append([max(log_likelihood_ratio)*10000000//10/1000000,'krm_plus1f',krm_plus1f[k],prefered_sens])


#-------------------------------------------------------------------re_krm_minus1f_sens_count_matrix

z=0
for l in re_krm_minus1f_sens_count_matrix:
    for j in range(len(l)):
        if l[j] == 0:
            z+=1
    if z>0:
        for j in range(len(l)):
            l[j]+=1
    z=0

for k in range(len(re_krm_minus1f_sens_count_matrix)):

    i=0
    log_likelihood_ratio= [0 for i in range(len(tag_types)*len(tag_types))]

    for m in range(len(re_krm_minus1f_sens_count_matrix[k])):
        for n in range(len(re_krm_minus1f_sens_count_matrix[k])):
            log_likelihood_ratio[i]= math.log10(re_krm_minus1f_sens_count_matrix[k][m]/re_krm_minus1f_sens_count_matrix[k][n])
            i+=1

    max_ratio_index=log_likelihood_ratio.index(max(log_likelihood_ratio))

    if (max_ratio_index%(len(tag_types)-1)==0):
        prefered_sens =tag_types[(max_ratio_index//len(tag_types))-1] # دومین 1-  به خاطر آنکه انديس ليست از صفر شروع مي شود کسر شده است 
    else:
        prefered_sens =tag_types[max_ratio_index//len(tag_types)]

    decision_list.append([max(log_likelihood_ratio)*10000000//10/1000000,'krm_minus1f',krm_minus1f[k],prefered_sens])

#-------------------------------------------------------------------re_krm_bi_sens_count_mtrix

z=0
for l in re_krm_bi_sens_count_mtrix:
    for j in range(len(l)):
        if l[j] == 0:
            z+=1
    if z>0:
        for j in range(len(l)):
            l[j]+=1
    z=0

for k in range(len(re_krm_bi_sens_count_mtrix)):
    
    i=0
    log_likelihood_ratio= [0 for i in range(len(tag_types)*len(tag_types))]

    for m in range(len(re_krm_bi_sens_count_mtrix[k])):
        for n in range(len(re_krm_bi_sens_count_mtrix[k])):
            log_likelihood_ratio[i]= math.log10(re_krm_bi_sens_count_mtrix[k][m]/re_krm_bi_sens_count_mtrix[k][n])
            i+=1

    max_ratio_index=log_likelihood_ratio.index(max(log_likelihood_ratio))

    if (max_ratio_index%(len(tag_types)-1)==0):
        prefered_sens =tag_types[(max_ratio_index//len(tag_types))-1] # دومین 1-  به خاطر آنکه انديس ليست از صفر شروع مي شود کسر شده است 
    else:
        prefered_sens =tag_types[max_ratio_index//len(tag_types)]

    decision_list.append([max(log_likelihood_ratio)*10000000//10/1000000,'krm_bif',krm_bif[k],prefered_sens])

#-------------------------------------------------------------------re_krm_tri_sens_count_mtrix

z=0
for l in re_krm_tri_sens_count_mtrix:
    for j in range(len(l)):
        if l[j] == 0:
            z+=1
    if z>0:
        for j in range(len(l)):
            l[j]+=1
    z=0

for k in range(len(re_krm_tri_sens_count_mtrix)):
    
    i=0
    log_likelihood_ratio= [0 for i in range(len(tag_types)*len(tag_types))]

    for m in range(len(re_krm_tri_sens_count_mtrix[k])):
        for n in range(len(re_krm_tri_sens_count_mtrix[k])):
            log_likelihood_ratio[i]= math.log10(re_krm_tri_sens_count_mtrix[k][m]/re_krm_tri_sens_count_mtrix[k][n])
            i+=1

    max_ratio_index=log_likelihood_ratio.index(max(log_likelihood_ratio))

    if (max_ratio_index%(len(tag_types)-1)==0):
        prefered_sens =tag_types[(max_ratio_index//len(tag_types))-1] # دومین 1-  به خاطر آنکه انديس ليست از صفر شروع مي شود کسر شده است 
    else:
        prefered_sens =tag_types[max_ratio_index//len(tag_types)]

    decision_list.append([max(log_likelihood_ratio)*10000000//10/1000000,'krm_trif',krm_trif[k],prefered_sens])

#######################################

from operator import itemgetter

sorted_decision_list=sorted(decision_list, key=itemgetter(0), reverse=True) # ليست را از بزرگ به کوچک بر حسب آيتم صفرم ليست سورت مي کند

#----------------------------------------

fp_train_sorted_decision_list = codecs.open('krm_4_train_sorted_decision_list.txt','w','utf_8')  #* اشاره گر به فايل متني که ليست تصميم در آن نوشته مي شود

#d=fp_train_sorted_decision_list.write('LogL مقدار')
#e=fp_train_sorted_decision_list.write('\t')
#f=fp_train_sorted_decision_list.write('نوع فيچر')
#g=fp_train_sorted_decision_list.write('\t\t')
#h=fp_train_sorted_decision_list.write('مقدار فيچر')
#i=fp_train_sorted_decision_list.write('\t')
#j=fp_train_sorted_decision_list.write('برچسب معني')
#k=fp_train_sorted_decision_list.write(u"\n")

for i in range(len(sorted_decision_list)):
    
    for j in range(len(sorted_decision_list[i])):
        a= fp_train_sorted_decision_list.write(str(sorted_decision_list[i][j]))
        b= fp_train_sorted_decision_list.write('\t')

    c= fp_train_sorted_decision_list.write(u"\n")

fp_train_sorted_decision_list.close()

