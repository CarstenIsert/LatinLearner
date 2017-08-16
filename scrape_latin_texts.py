from urllib.request import urlopen
from bs4 import BeautifulSoup


class LatinScraper:
    """Small script to help loading latin texts from the Latin Library
    You should only need to call process_main_site and then use manual 
    control to go through the different authors. 
    You can enter authors sites into the visited dict so that they will
    not be processed automatically.
    As this is a scraping tool, it depends heavily on the website and will
    probably need to change and adapt. That's the reason why I build in 
    manual control so that you can check what's going on.
    """
    LATIN_BASE_URL = 'http://thelatinlibrary.com/'
    DATA_DIRECTORY = './library/'
    VERBOSE = True
    visited = {'justin.html' : True, 'victor.html' : True, 'ammianus.html' : True, 'apuleius.html' : True, 'aug.html' : True, 'index.html' : True, 'classics.html' : True}
    book_counter = 1
    visited_list = []

    def remove_non_ascii_characters(self, text):
        cleaned_text = ''.join(character for character in text if ord(character) < 128)
        if len(cleaned_text) < 1000:
            print("ATTENTION: Probably wrong structure for document: ", filename)
            input("PLEASE CHECK Document")  
        return cleaned_text

    def generate_filename(self, title):
        # Clean characters from the title that are not allowed in filenames
        filename = ''.join(character for character in title if character.isalnum())
        filename = self.DATA_DIRECTORY + str(self.book_counter) + filename + '.txt'
        return filename

    def is_book_site(self, tables):
        """ To find out if this is a site containing text, there are either no tables in it,
        or if there are tables, then there should only be one and it has the first entry to
        index.html
        Seems there are more cases: If we are in a subdirectory, there is sometimes a table
        linking back to root and the author site where either the first or second entry is index.html.
        """
        if tables == []:
            return True
        table_HTML_references_a = tables[0].find_all('a')
        if table_HTML_references_a != []:
            if table_HTML_references_a[0]['href'] == 'index.html':
                return True
            if table_HTML_references_a[1]['href'] == 'index.html':
                return True
            if table_HTML_references_a[0]['href'][0] == '/':
                return True
        return False
    
    def scrape_book(self, book_soup):    
        book_text = book_soup.get_text()
        cleaned_book_text = self.remove_non_ascii_characters(book_text)
        book_title = book_soup.title.string
        filename = self.generate_filename(book_title)
        if self.VERBOSE: print("Processing and writing to file: ", filename)
        file = open(filename, "w")
        file.write(cleaned_book_text)
        file.close()
        self.book_counter += 1

    def process_table_site(self, tables, directory):  
        for table in tables:
            table_HTML_references_a = table.find_all('a')
            for item in table_HTML_references_a:
                if item['href'][0] == '/':
                    print("The first character is /, so skipping...", )
                    continue
                if self.VERBOSE: print("Looking at ", item)
                
                response = input("Should I process this item? (y/n/e(xit))?")
                if response == 'y':
                    self.process_reference(item, directory)
                elif response == 'e':
                    exit()
                else:
                    continue

    def process_reference(self, item, directory = ''):
        if self.VERBOSE: print("Processing item: ", directory, item['href'])
        if item['href'] in self.visited:
            if self.VERBOSE: print("==> Already visited!")
            return
    
        url = self.LATIN_BASE_URL + directory + item['href']
        try:
            website = urlopen(url)
            soup = BeautifulSoup(website, 'html.parser')
            next_tables = soup.find_all('table')
            if self.is_book_site(next_tables):
                self.scrape_book(soup)
            else:
                next_directory = directory + ('/').join(item['href'].split('/')[:-1]) 
                if next_directory != '':
                    next_directory += '/'
                    if self.VERBOSE: print(next_directory)
                     
                if self.VERBOSE: print("Found {} tables in this document.".format(len(next_tables)))
                response = input("Continue? y/n/e(xit)")
                
                if response == 'y':
                    self.process_table_site(next_tables, next_directory)
                elif response == 'e':
                    exit()
                else:
                    return
        except:
            if self.VERBOSE: print("Error opening this site...", url)
            return

    def process_main_site(self):  
        main_site = urlopen("http://thelatinlibrary.com")
        main_soup = BeautifulSoup(main_site, 'html.parser')
        tables = main_soup.find_all('table')
        #tables[1] contains the right information on the main page for the author table
        table_HTML_references_a = tables[1].find_all('a')
        if self.VERBOSE: print("Processing the Latin Library...")
        for author_ref in table_HTML_references_a:
            self.process_reference(author_ref)


my_scraper = LatinScraper()
my_scraper.process_main_site()