import requests
import pandas as pd
from bs4 import BeautifulSoup

from sgp4.api import Satrec

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
    z = []
    
    for row in soup:
        data = row.split('\n')

        if n==0:
            x.append(data[0])
            n = 1

        elif n==1:
            y.append(data[0])
            n = 2
            
        elif n==2:
            z.append(data[0])
            n = 0
    
    df = pd.DataFrame(columns=['name', 'l1', 'l2', 'r_vec', 'v_vec'])
    df.name = x
    df.l1 = y
    df.l2 = z
    
    r_vec = []
    v_vec = []
    for index, row in df.iterrows():
        sat = Satrec.twoline2rv(row['l1'], row['l2'])
        e, r, v = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
        r_vec.append(r)
        v_vec.append(v)
        
    df.r_vec = r_vec
    df.v_vec = v_vec
    
    print(df.r_vec[0])
            
            
if __name__ == "__main__":
    main()
 