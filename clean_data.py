import os
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
    end_of_text = raw_text.find("The Latin Library")
    output = raw_text[:end_of_text]
    return output
    
def lower_case(raw_text):
    output = raw_text.lower()
    return output

def remove_white_space(raw_text):
    output = ' '.join(raw_text.split())
    return output

def clean_text(raw_text):
    output = remove_end_note(raw_text)
    output = remove_brackets(output)
    output = remove_arabic_numbers(output)
    output = lower_case(output)
    output = remove_white_space(output)
    return output
    
def clean_directory(input_directory, output_directory):
    """ Runs through all the files in the specified directory assuming that those are files from the
    Latin Library and cleans the data with the set of defined methods above. As the format of the files
    is not very consistent, we clean only the major items.
    I decided to not remove latin numbers from the text to give the ML algorithm also some understanding
    about those numbers.
    """
    list_of_files = os.listdir(input_directory)
    for file_name in list_of_files:
        if file_name in ignore_list:
            continue
        print("Processing file: ", file_name)
        infile = open(input_directory + '/' + file_name)
        raw = infile.read()
        cleaned_text = clean_text(raw)
        infile.close()
        outfile = open(output_directory + '/' + file_name, 'w')
        outfile.write(cleaned_text)
        outfile.close()
    
    
def find_duplicates(directory):
    """ Finds duplicate files in the specified directory and also counts the number of unique files.
    The function prints the required remove commands, but does not actually delete the files.
    You can easily copy the rm commands to the console if you want to delete them.
    """
    content_list = {}
    list_of_files = os.listdir(directory)
    duplicate_count = 0
    file_count = 0
    for file_name in list_of_files:
        if file_name in ignore_list:
            continue
        file = open(directory + '/' + file_name)
        raw = file.read()
        hash_value = raw.__hash__()
        if hash_value in content_list:
            print('rm', file_name)
            duplicate_count += 1
        else:
            content_list[hash_value] = file_name
            file_count += 1
    
    print('Found duplicates: ', duplicate_count) 
    print('Found unique files: ', file_count)  
      
