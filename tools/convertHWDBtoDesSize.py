#coding=utf-8

"""
convert source file to destination file
format of destination file is
item,        type,             length,                instance,                        comment
Tag code,    char,             2B,                    "阿"=0xb0a1 stored as 0xa1b0
bitmap,      unsighed char,    width*height bytes,                                  ,  stored row by row


#####   HWDB1.1  start   ####
all character class is 3755
all character number is 1121749

test data all character number is:      223991
test data 1000 class,character number is 59688
test data 100 class,character number is 5975


train database,all character number is 897758
train data 1000 class, character number is  239064
train data 100 class,character number is 23936
#####   HWDB1.1  end   ####


#####   HWDB1.0  start   ####
all character class is 3740, all class in HWDB1.1

test data all character number is:        309684
test data 100 class,character number is:


train database,all character number is:   1246991
train data 100 class,character number is:
#####   HWDB1.0  end   ####


all db1.1 and db1.0 character number is:2678424


#####  compete test data start     ####
all character class is:3755

compete test data, all character number is:    224419
compete test data 100 class,character number is: 5988
#####  compete test data end     ####


all character
test data all 3755 class, character number is:   533675
train data all 3755 class, character number is: 2144749

test data 100 class, character number is:  14124
train data 100 class, character number is: 57304

characterTagcodeMapFile是一个字符编码和索引对照数组，
将汉字GBK编码保存到一个数组中，索引和编码形成一个影射关系

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from PIL import Image
import pickle
import OCRUtilv3s1

trainoridirnamev0 = "/home/allen/work/data/HWDB1.0orign/HWDB1.0trn_gnt"
traindesdirnamev0 = "/home/allen/work/data/HWDB1.0des64/HWDB1.0trn_gnt"

trainoridirname = "/home/allen/work/data/ORIdata/HWDB1.1trn_gnt"
traindesdirname = "/home/allen/work/data/HWDB1.1des64/HWDB1.1trn_gnt"

testoridirnamev0 = "/home/allen/work/data/HWDB1.0orign/HWDB1.0tst_gnt"
testdesdirnamev0 = "/home/allen/work/data/HWDB1.0des64/HWDB1.0tst_gnt"

testoridirname = "/home/allen/work/data/ORIdata/HWDB1.1tst_gnt"
testdesdirname = "/home/allen/work/data/HWDB1.1des64/HWDB1.1tst_gnt"

compete_test_oridir = "/home/allen/work/data/competition_test_data-gnt"
compete_test_desdir = "/home/allen/work/data/competeTestdes64"

descharacterTagcodeMapFile = "/home/allen/work/data/tagindexmap.pkl"


tag_buffer = []
bitmap_buffer = []
desfilename = ""

desCharSize = 64


def fromSrc2Des(oridirname,desdirname):
    """
    从原文件夹到目标文件夹转换，单个字符转换成指定大小的图片
    一个文件一个文件的转
    目标文件中每个字符由tagcode和bitmap组成,
    单个字符长度item length=tagcode(2) + bitmap(desCharSize*desCharSize)
    :param oridirname:
    :param desdirname:
    :return:
    """
    global tag_buffer,bitmap_buffer,desfilename

    orifilenames = sorted(os.listdir(oridirname))

    numOfFile = 1

    # if len(orifilenames)!=0:
    for filename in orifilenames:
        # filename = orifilenames[0]

        orifilename = os.path.join(oridirname, filename)
        desfilename = os.path.join(desdirname, filename)

        readFileContent(orifilename)
        write2DesFile(desfilename)

        print "%5dth:%s file convert completed" % (numOfFile, orifilename)
        numOfFile += 1


def readFileContent(filename):
    """
    读取原文件中单个字符数组，转换成目标大小后。
    将转换后数组(bitmap)和tagcode分开保存到bitmap_buffer和tag_buffer中
    :param filename: 原始文件
    :return:
    """
    global tag_buffer,bitmap_buffer,desCharSize
    with open(filename, mode='rb') as src_fobj:

        while src_fobj.read(1) !="":
            src_fobj.seek(-1,1)
            sampleSize = struct.unpack('<I', src_fobj.read(4))[0]
            # tag code is gbk
            tagCode = src_fobj.read(2)
            width = struct.unpack('<H', src_fobj.read(2))[0]
            height = struct.unpack('<H', src_fobj.read(2))[0]

            bitmap = np.zeros([height, width], dtype=np.float32)

            for i in xrange(height):
                for j in xrange(width):
                    pixel = struct.unpack('<B', src_fobj.read(1))[0]
                    bitmap[i, j] = pixel

            des = convertToDesSize(bitmap,desCharSize)

            tag_buffer.append(tagCode)
            bitmap_buffer.append(des)

            # print "%s"%tagCode.decode('gbk')


def write2DesFile(filename):
    """
    将bitmap_buffer和tag_buffer中的数据保存到目标文件中

    :param filename:目标文件
    :return:
    """
    global tag_buffer, bitmap_buffer,desCharSize

    with open(filename,mode='wb') as des_fobj:
        while len(tag_buffer) !=0:

            tagcode = tag_buffer.pop(0)
            bitmap = bitmap_buffer.pop(0)
            des_fobj.write(tagcode)
            for i in xrange(desCharSize):
                for j in xrange(desCharSize):
                    pixcel = struct.pack('<B', bitmap[i, j])
                    des_fobj.write(pixcel)

            # print "%s"%tagcode.decode('gbk')

def convertToDesSize(ori,desSize):
    """
    将原始单个字符数组中数字转换成目标大小数组返回
    :param ori: 原始字符数组
    :param desSize: 目标单个字符大小
    :return:
    """
    imgdata = Image.fromarray(ori)
    imgdata = imgdata.resize([desSize,desSize])
    des = np.array(imgdata)
    des = np.reshape(des,[desSize,desSize])
    return des.astype(np.ubyte)


def fun2():
    with open('test.ocr',mode='w') as fobj:
        fobj.write('nice to meet you')

    with open('/home/allen/work/data/HWDB1.1orign/HWDB1.1trn_gnt/1001-c.gnt',mode='rb') as fobj:
        size = fobj.read(4)
        print size
        tag = fobj.read(2)
        print '%r'%tag


def showOriImage(dirname):
    orifilenames = sorted(os.listdir(dirname))

    filename = orifilenames[0]
    filename = os.path.join(dirname, filename)

    print filename

    with open(descharacterTagcodeMapFile) as fobj:
        tagcodeMap = pickle.load(fobj)

    with open(filename,mode='rb') as fobj:
        for numi in range(5):


            # for numj in range(6):

            sampleSize = struct.unpack('<I',fobj.read(4))[0]
            tagcode = fobj.read(2)
            width = struct.unpack('<H', fobj.read(2))[0]
            height = struct.unpack('<H', fobj.read(2))[0]

            bitmap = np.zeros([height,width],dtype=np.float32)

            for i in xrange(height):
                for j in xrange(width):
                    pixel = struct.unpack('<B', fobj.read(1))[0]
                    bitmap[i, j] = pixel

            print tagcode.decode('gbk')


            if tagcode in tagcodeMap:

                img = Image.fromarray(bitmap)
                plt.figure()
                # plt.subplot(2,3, numj+1)
                plt.imshow(img)
                # print

                plt.show()


def showDesImage(desdirname):
    global desCharSize
    desFilenames = sorted(os.listdir(desdirname))

    filename = desFilenames[5]
    filename = os.path.join(desdirname, filename)

    print filename

    with open(filename,mode='rb') as fobj:
        itemSize = 2+desCharSize*desCharSize
        content = fobj.read()
        for i in xrange(5):

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



def calculatCharCount(dirname):
    """
    计算原文件夹下所有不同的字符个数
    :return:
    """
    orifilenames = sorted(os.listdir(dirname))

    charSet = set()
    numOfFile = 1


    for filename in orifilenames:
        filename = os.path.join(dirname, filename)

        with open(filename, mode='rb') as src_fobj:

            while src_fobj.read(1) != "":
                src_fobj.seek(-1, 1)
                src_fobj.read(4)
                # tag code is gbk
                tagCode = src_fobj.read(2)

                # print tagCode.decode('gbk')

                width = struct.unpack('<H', src_fobj.read(2))[0]
                height = struct.unpack('<H', src_fobj.read(2))[0]

                for i in xrange(height):
                    for j in xrange(width):
                        pixel = src_fobj.read(1)

                if tagCode not in charSet:
                    charSet.add(tagCode)
        print "%3d now character size is "%numOfFile,len(charSet)
        numOfFile += 1

def calculateDesAllCharacterCount(dirname,tagcodefile = None):
    """
    计算目标文件夹下所有文件字符总数,将不同字符的tagcode保存到tagcodefile文件中
    显示单个文件字符数，总字符数,该目录下不同字符总数
    """
    global desCharSize
    filenames = sorted(os.listdir(dirname))

    allCharacterCount = 0
    filenum = 0

    chSize = 2+desCharSize*desCharSize
    difftagcodes = []

    # if len(filenames) != 0:
    #     filename = filenames[0]
    #     filename = os.path.join(dirname,filename)

    for each in filenames:
        filename = os.path.join(dirname, each)

        with open(filename,mode='rb') as fobj:
            content = fobj.read()
            filenum += 1
            fileCharacterCount = len(content)/chSize
            for iti in range(fileCharacterCount):
                start = iti*chSize
                tagcode = content[start:start+2]
                if tagcode not in difftagcodes:
                    difftagcodes.append(tagcode)


            allCharacterCount += fileCharacterCount

            print "%d th file, character count is %d, all count is %d"%(filenum, fileCharacterCount, allCharacterCount)

    if tagcodefile is not None:
        with open(tagcodefile,mode='w') as wfobj:
            pickle.dump(difftagcodes,wfobj)
    print "in "+ dirname + " dir, diff character number is ",len(difftagcodes)



def createTagIndexMap():
    """
    将每个汉字编码存放到列表中，按照读入的先后顺序添加
    :return:
    """
    charlist = []
    filenames = sorted(os.listdir(traindesdirnamev0))

    numOfFile = 0
    itemlength = 64*64+2

    while len(charlist) < 3740:
        filename = filenames[numOfFile]
        filename = os.path.join(traindesdirnamev0, filename)

        with open(filename,mode='rb') as fobj:
            numOfFile += 1
            content = fobj.read()

            charnums = len(content)/itemlength
            for i in xrange(charnums):

                start = i * (itemlength)
                tagcode = content[start:start+2]

                if tagcode not in charlist:
                    charlist.append(tagcode)

    filenames = sorted(os.listdir(traindesdirname))
    while len(charlist) < 3755:
        filename = filenames[numOfFile]
        filename = os.path.join(traindesdirname, filename)

        with open(filename,mode='rb') as fobj:
            numOfFile += 1
            content = fobj.read()

            charnums = len(content)/itemlength
            for i in xrange(charnums):

                start = i * (itemlength)
                end = start + 2
                tagcode = content[start:end]

                if tagcode not in charlist:
                    charlist.append(tagcode)

    with open(descharacterTagcodeMapFile, mode='w') as fobj:
        pickle.dump(charlist,fobj)

    print "create tagcode index map file done"


def create_diff_two_DB(oridirname):
    """

    :param oridirname:
    :return:
    """
    with open(descharacterTagcodeMapFile) as fobj:
        db1tagcodemap = pickle.load(fobj)

    db0tagcodemap = []
    orifilenames = sorted(os.listdir(oridirname))

    characternum = 0

    for filename in orifilenames:
        filename = os.path.join(oridirname, filename)

        with open(filename, mode='rb') as src_fobj:

            while src_fobj.read(1) != "":
                characternum += 1

                src_fobj.seek(-1, 1)
                src_fobj.read(4)
                # tag code is gbk
                tagCode = src_fobj.read(2)

                if tagCode not in db0tagcodemap:
                    db0tagcodemap.append(tagCode)
                if tagCode not in db1tagcodemap:
                    print tagCode.decode('gbk')," not in db1"

                if characternum % 3000 == 0:
                    print "now db0 tagcodemap file size is ",len(db0tagcodemap)

                # print tagCode.decode('gbk')

                width = struct.unpack('<H', src_fobj.read(2))[0]
                height = struct.unpack('<H', src_fobj.read(2))[0]

                for i in xrange(height):
                    for j in xrange(width):
                        pixel = src_fobj.read(1)

    print "db0 all character is ",characternum
    with open(descharacterTagcodeMapFile, mode='w') as fobj:
        pickle.dump(db0tagcodemap,fobj)


def convertdb0_ori_to_des(orifilename,desdir,startindex):
    """
    将HWDB1.0中原数据文件转换成目标大小的以3000个字为一组，存储到单个文件中。

    :param orifilename:
    :param desdir:
    """
    global bitmap_buffer,tag_buffer,desCharSize

    desfile_contain_char_num = 3000
        #des file name is tart 0001, means version 1.0, index start 001,train des file num is 416
    desfile_index = startindex

    with open(orifilename, mode='rb') as src_fobj:


        while src_fobj.read(1)!='':
            src_fobj.seek(-1,1)
            sampleSize = struct.unpack('<I', src_fobj.read(4))[0]
            # tag code is gbk
            tagCode = src_fobj.read(2)
            width = struct.unpack('<H', src_fobj.read(2))[0]
            height = struct.unpack('<H', src_fobj.read(2))[0]

            bitmap = np.zeros([height, width], dtype=np.float32)

            for i in xrange(height):
                for j in xrange(width):
                    pixel = struct.unpack('<B', src_fobj.read(1))[0]
                    bitmap[i, j] = pixel

            des = convertToDesSize(bitmap, desCharSize)

            tag_buffer.append(tagCode)
            bitmap_buffer.append(des)

            if len(tag_buffer) == desfile_contain_char_num:
                desfile_name = "{:0>4}".format(str(desfile_index)) + "-c.gnt"
                desfile_name = os.path.join(desdir, desfile_name)

                write2DesFile(desfile_name)
                print desfile_name,"already create"
                desfile_index += 1

    if len(tag_buffer)!=0:
        desfile_name = str(desfile_index) + "-c.gnt"
        desfile_name = os.path.join(desdir, desfile_name)
        write2DesFile(desfile_name)

    print "process over"


def convtDB0():
    global traindesdirnamev0,testdesdirnamev0

    orifilename = '/media/allen/CEFC86D4FC86B5ED/学习资料/机器学习/OCR/isolated_data/1.0trn-c.gnt'
    convertdb0_ori_to_des(orifilename, traindesdirnamev0, 1)

    orifilename = '/media/allen/CEFC86D4FC86B5ED/学习资料/机器学习/OCR/isolated_data/1.0tst-c.gnt'
    convertdb0_ori_to_des(orifilename, testdesdirnamev0, 417)


def convertcompeteDB2deslimit(competeoridirname,competedesdirname):
    global desCharSize
    # competeoridirname = "/home/allen/work/data/competeTestdes64"
    # competedesdirname = "/home/allen/work/data/cmptTestdes64limit"
    limitnum = 2000
    orifilenames = sorted(os.listdir(competeoridirname))
    startindex = 1
    itemLength = 2 + desCharSize*desCharSize

    content = ""

    for i in range(len(orifilenames)):
        orifilename = os.path.join(competeoridirname,orifilenames[i])
        print "start process",orifilename

        rfobj = open(orifilename,mode='rb')
        if len(content) == 0:
            content = rfobj.read()
        else:
            content += rfobj.read()
        rfobj.close()

        while len(content)/itemLength>=limitnum:
            desfilename = "c" + "{:0>4}".format(startindex) + ".gnt"
            desfilename = os.path.join(competedesdirname, desfilename)
            tcontent = content[:limitnum*itemLength]
            content = content[limitnum*itemLength:]
            with open(desfilename,mode='wb') as wfobj:
                wfobj.write(tcontent)
            startindex += 1

    if len(content) != 0:
        desfilename = "c" + "{:0>4}".format(startindex) + ".gnt"
        desfilename = os.path.join(competedesdirname, desfilename)
        with open(desfilename, mode='wb') as wfobj:
            wfobj.write(content)
    print "process done"


def convertTrainDB2Deslimit():
    """
    将转换后的数据库文件夹中所有文件读取，按照单文件字符数存储到目标文件夹中
    :return:
    """
    global desCharSize

    itemLength = 2 + desCharSize * desCharSize

    traindirnames = ["/home/allen/work/data/competeTestdes64"]
    desdirname = "/home/allen/work/data/singleFileLimitNum/competeData"
    desfilecharnums = 2000
    cur_file_index=1
    cur_dir_index = 1

    dirname = os.path.join(desdirname, "{:0>2}".format(str(cur_dir_index)))
    os.mkdir(dirname)

    content = ""

    for eachdir in traindirnames:
        orifilenames = sorted(os.listdir(eachdir))
        orifilenames = [os.path.join(eachdir,eachfile) for eachfile in orifilenames]

        for eachfile in orifilenames:
            print "start process",eachfile

            rfobj = open(eachfile, mode='rb')
            if len(content) == 0:
                content = rfobj.read()
            else:
                content += rfobj.read()
            rfobj.close()

            while len(content) / itemLength >= desfilecharnums:
                desfilename = OCRUtilv3s1.get_filename(desdirname,cur_dir_index,cur_file_index)

                tcontent = content[:desfilecharnums * itemLength]
                content = content[desfilecharnums * itemLength:]
                with open(desfilename, mode='wb') as wfobj:
                    wfobj.write(tcontent)

                if cur_file_index%10 ==0:
                    cur_dir_index += 1
                    dirname = os.path.join(desdirname, "{:0>2}".format(str(cur_dir_index)))
                    os.mkdir(dirname)
                cur_file_index += 1

    if len(content) != 0:
        desfilename = OCRUtilv3s1.get_filename(desdirname,cur_dir_index,cur_file_index)
        with open(desfilename, mode='wb') as wfobj:
            wfobj.write(content)
    print "process done"





def fun3():
    db1tagcodefile = "/home/allen/work/data/tagindexmapdb1.pkl"
    db0tagcodefile = "/home/allen/work/data/tagindexmapdb0.pkl"
    competetagcodefile = "/home/allen/work/data/tagindexmapcompete.pkl"
    descharacterTagcodeMapFile = "/home/allen/work/data/tagindexmap.pkl"

    with open(db0tagcodefile) as rfobj:
        db0tags = pickle.load(rfobj)

    with open(db1tagcodefile) as rfobj:
        db1tags = pickle.load(rfobj)

    with open(competetagcodefile) as rfobj:
        comtags = pickle.load(rfobj)

    with open(descharacterTagcodeMapFile) as rfobj:
        destags = pickle.load(rfobj)

    #字符在db0 不在 db1
    for each in db0tags:
        if each not in db1tags:
            print each.decode('gbk'),"in db0, not in db1"

    print "mission 1 done"

    # 字符在compete db 不在 db1
    for each in comtags:
        if each not in db1tags:
            print each.decode('gbk'),'in compete db, not in db1'
    print "mission 2 done"

    #字符在db1 不在 desdb
    for each in db1tags:
        if each not in destags:
            print each.decode('gbk'),'in db1, not in desdb'
    print 'mission 3 done'









def test():
    db1tagcodefile = "/home/allen/work/data/tagindexmapdb1.pkl"
    db0tagcodefile = "/home/allen/work/data/tagindexmapdb0.pkl"
    competetagcodefile = "/home/allen/work/data/tagindexmapcompete.pkl"

    competelimitdirname = "/home/allen/work/data/cmptTestdes64limit"

    traindirname = "/home/allen/work/data/trainDB/tempdir2/1"

    # fromSrc2Des(compete_test_oridir,compete_test_desdir)


    # convertTrainDB2Deslimit()

    # convertcompeteDB2deslimit(compete_test_desdir,competelimitdirname)
    # fun3()
    # createTagIndexMap()

    dirname = "/home/allen/work/data/singleFileLimitNum/competeData/12"
    calculateDesAllCharacterCount(dirname)
    # fromSrc2Des(compete_test_oridir,compete_test_desdir)
    # calculatCharCount(testoridirnamev0)
    # create_diff_two_DB(testoridirnamev0)

    # convtDB0()



    # showOriImage(testoridirname)
    # showDesImage(dirname)

if __name__ == "__main__":
    test()