import numpy as np

def getseandlabels(file_path):
    se_trigger_entity=open(file_path,encoding='utf-8').readlines()
    ses=[]
    trigger=[]
    entity=[]
    for i in range(len(se_trigger_entity)-2):
        if (i)%3==0:
            ses.append(se_trigger_entity[i].strip().split(' '))
            trigger.append(se_trigger_entity[i+1].strip().split(' '))
            entity.append(se_trigger_entity[i+2].strip().split(' '))
    return  ses,trigger,entity

def getall(text,trigger_labels,entity_labels):
    train_sequences=[]
    train_positions=[]
    train_entitys=[]
    train_labels=[]
    number=np.zeros(34)
    for j in range(len(text)):#sentences
        ses=[]
        trs=[]
        ens=[]
        pos=[]
        for i in range(15,len(text[j])-15):
            se=text[j][i-15:i+16]
            ses.append(se)
            labels34=[0,0,0,0,0,0,0,0,0,0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0,0,0,0]
            tr=trigger_labels[j][i]   #34lei
            labels34[int(tr)]=1
            number[int(tr)]=number[int(tr)]+1
            trs.append(labels34)
            en=entity_labels[j][i-15:i+16]
            ens.append(en)
            pos.append([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
        train_sequences.append(ses)
        train_labels.append(trs)
        train_entitys.append(ens)
        train_positions.append(pos)
    print(list(number))
    return train_sequences,train_positions,train_entitys,train_labels
def addnone(text,trigger,entity):
    sentences_addnone=[]
    triggers_addnone=[]
    entitys_addnone=[]
    for tt in text:
        tt_addnone=['None','None','None','None','None','None','None','None','None','None',
                    'None','None','None','None','None']
        for ttt in tt:
            tt_addnone.append(ttt)
        for i in range(15):
            tt_addnone.append('None')
        sentences_addnone.append(tt_addnone)
    for tt in trigger:
        tt_addnone=['0','0','0','0','0','0','0','0','0','0',
                    '0','0','0','0','0']
        for ttt in tt:
            tt_addnone.append(ttt)
        for i in range(15):
            tt_addnone.append('0')
        triggers_addnone.append(tt_addnone)
    for tt in entity:
        tt_addnone=['14','14','14','14','14','14','14','14','14','14',
                    '14','14','14','14','14']
        for ttt in tt:
            tt_addnone.append(ttt)
        for i in range(15):
            tt_addnone.append('14')
        entitys_addnone.append(tt_addnone)

    return sentences_addnone,triggers_addnone,entitys_addnone

def predata():

   #get sequences/trigger label/typelabel  each sentence
   #What,does,that,have,to,do,with,the,war,in,Iraq,?
   #0,0,0,0,0,0,0,0,13,0,0,0
   # 0,0,0,0,0,0,0,0,0,0,2,0
   train_sentences,train_trigger_labels,train_entity_labls=getseandlabels("train_new.txt")
   test_sentences, test_trigger_labels, test_entity_labls = getseandlabels("test_new.txt")
   dev_sentences,dev_trigger_labels, dev_entity_labls = getseandlabels("dev_new.txt")
   print(len(train_sentences),len(test_sentences),len(dev_sentences))
   #for i in range(len(train_sentences)):
       #for j in range(len(train_sentences[i])):
           #train_sentences[i][j]=train_sentences[i][j].lower()
  # for i in range(len(test_sentences)):
       #for j in range(len(test_sentences[i])):
          # test_sentences[i][j] = test_sentences[i][j].lower()
  # for i in range(len(dev_sentences)):
      # for j in range(len(dev_sentences[i])):
         #  dev_sentences[i][j] = dev_sentences[i][j].lower()



   ## add text addlabel addtype
   train_sentences_add,train_trigger_labels_add,train_entity_labls_add=addnone(train_sentences,train_trigger_labels,train_entity_labls)
   test_sentences_add, test_trigger_labels_add, test_entity_labls_add =addnone(test_sentences, test_trigger_labels, test_entity_labls)
   dev_sentences_add,dev_trigger_labels_add, dev_entity_labls_add = addnone(dev_sentences,dev_trigger_labels, dev_entity_labls)

   train_sequences,train_positions,train_entitys,train_labels=getall(train_sentences_add,train_trigger_labels_add,train_entity_labls_add)
   test_sequences, test_positions, test_entitys, test_labels = getall(test_sentences_add, test_trigger_labels_add,
                                                                      test_entity_labls_add)
   dev_sequences,dev_positions,dev_entitys, dev_labels = getall(dev_sentences_add, dev_trigger_labels_add,
                                                                dev_entity_labls_add)
  # print(train_sequences[16][5],train_positions[16][5],train_entitys[16][5],train_labels[16][5])

 #  train_sequences,train_positions,train_entitys,train_labels=getwordall(train_sentences,train_trigger_labels,train_entity_labls)
   return train_sentences,test_sentences,dev_sentences,train_trigger_labels,test_trigger_labels,dev_trigger_labels,train_entity_labls,test_entity_labls,dev_entity_labls,\
          train_sequences,test_sequences,dev_sequences,\
          train_positions,test_positions,dev_positions,\
          train_entitys,test_entitys,dev_entitys, \
          train_labels, test_labels, dev_labels

