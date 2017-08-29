# TODO: Remove the numbers at the beginning
# TODO: Remove The Latin Library at the end
# TODO: Remove The Classics Page at the end

import os
import nltk

ignore_list = ['.DS_Store']

def tokenize():
  file = open('library/1Catullus.txt')
  raw = file.read()
  tokens = nltk.word_tokenize(raw)
  print(tokens)
  text = nltk.Text(tokens)
  text.collocations()
  endOfText = raw.find("The Latin Library")
  print(endOfText)


def find_duplicates(directory):
  """ Finds duplicate files in the specified directory and also counts the number of unique files.
  The function prints the required remove commands, but does not actually delete the files.
  You can easily copy the rm commands to the console if you want to delete them.
  """
  contentList = {}
  listOfFiles = os.listdir(directory)
  duplicateCount = 0
  fileCount = 0
  for fileName in listOfFiles:
    if fileName in ignore_list:
        continue
    file = open(directory + '/' + fileName)
    raw = file.read()
    hashValue = raw.__hash__()
    if hashValue in contentList:
      print('rm', fileName)
      duplicateCount += 1
    else:
      contentList[hashValue] = fileName
      fileCount += 1
  
  print('Found duplicates: ', duplicateCount) 
  print('Found unique files: ', fileCount)  
      
find_duplicates('./library')

