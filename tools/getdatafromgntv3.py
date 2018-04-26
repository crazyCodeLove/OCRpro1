#coding=utf-8

"""
drop sample technology
目标文件中每个字符由tagcode和bitmap组成,
单个字符长度item length=tagcode(2) + bitmap(desCharSize*desCharSize)

#####   HWDB1.1  start   ####
all character class is 3755

test data all character number is:      223991
test data 1000 class,character number is 59688
test data 100 class,character number is 5975


train database,all character number is 897758
train data 1000 class, character number is  239064
train data 100 class,character number is 23936
#####   HWDB1.1  end   ####


#####   HWDB1.0  start   ####
all character class is 3740, all class in HWDB1.1

test data all character number is:       309684
test data 100 class,character number is:


train database,all character number is:  1246991
train data 100 class,character number is:
#####   HWDB1.0  end   ####

all character

test data 100 class, character number is:  14202
train data 100 class, character number is: 56987

"""

import os
import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import random


traindirnames = "/home/allen/work/data/limitclass/train100class"
testdirname = "/home/allen/work/data/limitclass/test100class"

descharacterTagcodeMapFile = "/home/allen/work/data/100class.pkl"

charWidth = 64
itemLength = charWidth*charWidth + 2

def next_batch_dirs(batchnum, dirnames, charWidth, character_class, charTagcodeMapFile):
    """

    :param batchnum: 每次取的字符数
    :param dirnames: 目标文件的文件夹列表
    :param itemLength: 每个字符占用的字节数, = tagcode(2) + charWidth(64)*charWidth(64)
    :param character_class: 目标文件夹下的字符类别数
    :param charTagcodeMapFile: 使用哪一个tagcodemap影射文件
    :return: one hot 编码的一组[batch_x,batch_y]
    """

    itemLength = 2+charWidth*charWidth
    filenames = []
    for eachdir in dirnames:
        tfnames = os.listdir(eachdir)
        tfnames = [os.path.join(eachdir,fname) for fname in tfnames]
        filenames.extend(tfnames)

    random.shuffle(filenames)
    filenum = -1
    batch_x = []
    batch_y = []
    oricontent = []

    while True:
        filenum += 1
        filenum = filenum % len(filenames)

        filename = filenames[filenum]

        # print filename

        with open(filename,mode='rb') as fobj:
            content = fobj.read()
            contentlength = len(content)
            start = 0

            while start<contentlength:
                if len(batch_y) == batchnum:
                    batch_x = []
                    batch_y = []
                    oricontent = []

                fetchnum = batchnum - len(batch_x)
                end = start+ fetchnum * itemLength

                if end <= contentlength:
                    data2list(content, start, end, batch_x, batch_y, oricontent, itemLength, charTagcodeMapFile)
                    start = end
                    batch_x,batch_y = fromList2Stand(batch_x,batch_y,character_class)
                    oricontent = "".join(oricontent)
                    yield batch_x,batch_y,oricontent


                else:
                    end = contentlength
                    data2list(content, start, end, batch_x, batch_y, oricontent, itemLength, charTagcodeMapFile)
                    start = contentlength



def fromList2Stand(batch_x,batch_y,character_class):
    """

    :param batch_x:
    :param batch_y:
    :param character_class:
    :return:
    """
    out_x = 255 - np.array(batch_x).astype(np.float32)

    out_y = np.zeros([len(batch_y),character_class],dtype=np.float64)
    for i in xrange(len(batch_y)):
        out_y[i,batch_y[i]] = 1.0

    return out_x,out_y


def data2list(data,start,end,batch_x,batch_y, oricontent,itemLength,characterTagcodeMapFile):
    """
    :param characterTagcodeMapFile:
    """
    length = (end-start) / itemLength
    oricontent.append(data[start:end])


    with open(characterTagcodeMapFile) as fobj:
        tagmap = pickle.load(fobj)

    for i in xrange(length):
        substart = i * itemLength + start
        tagcode = data[substart:substart+2]
        bitmap = data[substart+2:substart + itemLength]


        bitmap = [struct.unpack('<B',pixcel)[0] for pixcel in bitmap]

        batch_y.append(tagmap.index(tagcode))
        batch_x.append(bitmap)




def fun1():
    global charWidth, itemLength, descharacterTagcodeMapFile
    character_class = 100
    filename = "/home/allen/work/data/trainDB/tempdir1/new.gnt"


    number = 6
    gen = next_batch_dirs(number, [testdirname, ], charWidth, character_class, descharacterTagcodeMapFile)

    with open(descharacterTagcodeMapFile) as fobj:
        tagmap = pickle.load(fobj)

    for j in xrange(2):
        x, y, oridata = gen.next()
        plt.figure()

        for i in xrange(number):
            pic = x[i, :]
            label = y[i, :]
            index = np.argmax(label)
            print tagmap[index].decode('gbk'),

            pic = np.reshape(pic, [charWidth, charWidth])
            plt.subplot(2, 3, i + 1)
            plt.imshow(pic)
        print

        writeData2File(filename,oridata)


        plt.show()


def writeData2File(filename,data):
    if not os.path.exists(filename):
        with open(filename,mode='wb') as wfobj:
            wfobj.write(data)
    else:
        with open(filename,mode='ab') as wfobj:
            wfobj.write(data)

def getCharNumsInDir(dirname,charWidth):
    """

    :param dirname: 单个目录文件夹下字符数目
    :param charWidth:
    :return:
    """
    itemLength = 2+charWidth*charWidth
    filenames = os.listdir(dirname)
    filenames = [os.path.join(dirname,each) for each in filenames]

    totalnums = 0
    for eachfile in filenames:
        with open(eachfile,mode='rb') as rfobj:
            content = rfobj.read()
            totalnums += len(content)/itemLength
    return  totalnums

def getCharNumsInFile(filename,charWidth):
    itemLength = 2 + charWidth*charWidth
    if not os.path.exists(filename):
        return 0

    with open(filename,mode='rb') as rfobj:
        content = rfobj.read()
        totalnums = len(content)/itemLength
    return totalnums


def showDesImage(desdirname):
    desCharSize = 64
    desFilenames = sorted(os.listdir(desdirname))

    filename = desFilenames[0]
    filename = os.path.join(desdirname, filename)

    print filename

    with open(filename,mode='rb') as fobj:
        itemSize = 2+desCharSize*desCharSize
        content = fobj.read()
        for i in xrange(10):

            start = i*itemSize
            end = (i+1)*itemSize
            tagcode = content[start:start+2]
            bitmap = content[start+2:end]
            bitmap = [struct.unpack('<B',each)[0] for each in bitmap]
            bitmap = np.array(bitmap).astype(np.ubyte).reshape([desCharSize,desCharSize])

            print tagcode.decode('gbk')
            plt.figure()
            img = Image.fromarray(bitmap)
            plt.imshow(img)

            plt.show()



def test():
    # fun1()

    desdirname = "/home/allen/work/data/trainDB/tempdir1"
    showDesImage(desdirname)

    print "done"









if __name__ == "__main__":
    test()
