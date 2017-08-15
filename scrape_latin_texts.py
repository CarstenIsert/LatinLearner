from urllib.request import urlopen
from bs4 import BeautifulSoup

LATIN_BASE_URL = 'http://thelatinlibrary.com/'
DATA_DIRECTORY = './library/'
VERBOSE = True
visited = {'victor.html' : True, 'ammianus.html' : True, 'apuleius.html' : True, 'aug.html' : True, 'index.html' : True, 'classics.html' : True}
book_counter = 1


def remove_non_ascii_characters(text):
  cleaned_text = ''.join(character for character in text if ord(character) < 128)
  if len(cleaned_text) < 1000:
      print("ATTENTION: Probably wrong structure for document: ", filename)
      input("PLEASE CHECK Document")  
  return cleaned_text

def generate_filename(title):
    # Clean characters from the title that are not allowed in filenames
    filename = ''.join(character for character in title if character.isalnum())
    filename = DATA_DIRECTORY + str(book_counter) + filename + '.txt'
    return filename

def is_book_site(tables):
    """ To find out if this is a site containing text, there are either no tables in it,
    or if there are tables, then there should only be one and it has the first entry to
    index.html
    Seems there are more cases: If we are in a subdirectory, there is sometimes a table
    linking back to root and the author site.
    """
    if tables == []:
        return True
    table_HTML_references_a = tables[0].find_all('a')
    if table_HTML_references_a != []:
        if table_HTML_references_a[0]['href'] == 'index.html':
            return True
        if table_HTML_references_a[0]['href'][0] == '/'
            return True
    return False
    
def scrape_book(book_soup):    
    book_text = book_soup.get_text()
    cleaned_book_text = remove_non_ascii_characters(book_text)
    book_title = book_soup.title.string
    filename = generate_filename(book_title)
    if VERBOSE: print("Processing and writing to file: ", filename)
    file = open(filename, "w")
    file.write(cleaned_book_text)
    file.close()
    
    global book_counter
    book_counter += 1

def process_table_site(tables, directory):  
    for table in tables:
        table_HTML_references_a = table.find_all('a')
        for item in table_HTML_references_a:
            if VERBOSE: print("Looking at ", item)
            response = input("Should I process this item? (y/n/e(xit))?")
            if response == 'y':
                process_reference(item, directory)
            elif response == 'e':
                exit()
            else:
                continue

def process_reference(item, directory = ''):
    if VERBOSE: print("Processing item: ", directory, item['href'])
    if item['href'] in visited:
        if VERBOSE: print("==> Already visited!")
        return

    url = LATIN_BASE_URL + directory + item['href']
    website = urlopen(url)
    soup = BeautifulSoup(website, 'html.parser')
    next_tables = soup.find_all('table')
    if is_book_site(next_tables):
        scrape_book(soup)
    else:
        next_directory = directory + ('/').join(item['href'].split('/')[:-1]) 
        if next_directory != '':
            next_directory += '/'
            if VERBOSE: print(next_directory)
             
        if VERBOSE: print("Found {} tables in this document.".format(len(next_tables)))
        response = input("Continue? y/n/e(xit)")
        
        if response == 'y':
            process_table_site(next_tables, next_directory)
        elif response == 'e':
            exit()
        else:
            return

def process_main_site():  
    main_site = urlopen("http://thelatinlibrary.com")
    main_soup = BeautifulSoup(main_site, 'html.parser')
    tables = main_soup.find_all('table')
    #tables[1] contains the right information on the main page for the author table
    table_HTML_references_a = tables[1].find_all('a')
    if VERBOSE: print("Processing the Latin Library...")
    for author_ref in table_HTML_references_a:
        process_reference(author_ref)

process_main_site()