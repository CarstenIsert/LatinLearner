import os
import nltk
import re
from string import digits

ignore_list = ['.DS_Store']

def remove_arabic_numbers(raw_text):
  remove_digits = str.maketrans('', '', digits)
  output = raw_text.translate(remove_digits)
  return output
  
def remove_brackets(raw_text):
  output = re.sub("[\(\[].*?[\)\]]", "", raw_text)
  return output

def remove_end_note(raw_text):
  endOfText = raw_text.find("The Latin Library")
  output = raw_text[:endOfText]
  return output
    
def tokenize(raw_text):
  tokens = nltk.word_tokenize(raw_text)
  text = nltk.Text(tokens)
  return text

def clean_directory(input_directory, output_directory):
  """ Runs through all the files in the specified directory assuming that those are files from the
  Latin Library and cleans the data with the set of defined methods above. As the format of the files
  is not very consistent, we clean only the major items.
  I decided to not remove latin numbers from the text to give the ML algorithm also some understanding
  about those numbers.
  """
  listOfFiles = os.listdir(input_directory)
  for fileName in listOfFiles:
    if fileName in ignore_list:
        continue
    print("Processing file: ", fileName)
    infile = open(input_directory + '/' + fileName)
    raw = infile.read()
    infile.close()
    raw = remove_end_note(raw)
    raw = remove_brackets(raw)
    raw = remove_arabic_numbers(raw)
    outfile = open(output_directory + '/' + fileName, 'w')
    outfile.write(raw)
    outfile.close()
    
    
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
      
find_duplicates('library')
clean_directory('library', 'clean_library')
