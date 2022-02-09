import xml.etree.ElementTree as ET
import os

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)


xmlFileList = []
xmlpath= r"/mnt/Gu/VOCdevkit/VOC2007/Annotations/"
ic=0
allFilePath(xmlpath,xmlFileList)
for temp in xmlFileList:
    try:
        ic+=1
        target = ET.parse(temp).getroot()
        # print(ic,temp)
    except:
        print(ic,temp)