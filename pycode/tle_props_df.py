import requests
import pandas as pd
from bs4 import BeautifulSoup

def main():
    
    # Scrape data 
    for url in ['https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=starlink&FORMAT=tle']:
        try:
            raw = requests.get(url)
        except Exception as err:
            print(f'Error: {err}')
        else:
            print('Scraping TLEs..')
            
    soup = BeautifulSoup(raw.content, 'html.parser')
    soup = str(soup.get_text()).splitlines()
    
    n = 0
    x = []
    y = []
    
    for row in soup:
        data = row.split(' ')

        if n==0:
            x.append(data[0])

        elif ((n==1) | (n==2)):
            data = list(filter(None, data))
            x.extend(data[1:])    

        n+=1
        if n>2:
            y.append(x)
            n = 0
            x = []

    cols = ['name', 'sat_num1', 'int_desig', 'epoch','1d_mm', 
            '2d_mm', 'bstar', 'eph_type', 'checksum1', 'sat_num2', 
            'inc', 'raan', 'ecc', 'argp', 'ma', 'mm', 'checksum2']
    
    return pd.DataFrame(y, columns=cols)
            
            
if __name__ == "__main__":
    main()
 