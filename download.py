import requests
from lxml import etree

base_url = 'https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/Eurasia/'


def zip_urls(base_url):
    res = requests.get(base_url)
    r_text = res.text
    html_text =etree.HTML(r_text)
    lis = html_text.xpath("//ul/li/a/text()")
    
    lis = [i.replace(" ","") for i in lis if ((i[1]=="N") and (i[4]=="E"))]
    lis = [i for i in lis 
           if (int(i[1:3]) in list(range(15,61))) and (int(i[4:7]) in list(range(72,136)))
           ]
    lis = [base_url+i for i in lis]
    return lis

def get_zip(base_url,zip_list):
    """
    Parameters:
        base_url--->
        zip_lists--->
    
    Returns:
        None
    """
    import os
    import zipfile
    path = r"C:\Users\23909\Desktop\zip_data"
    for url in zip_list:
        res = requests.get(url,stream=True)
        print("*"*20,"正在获取{}".format(url))
        with open((os.path.join(path,url.split('/')[-1]+".txt")), 'wb') as f:
            for line in res.iter_lines():
                if line:
                    f.write(line)
                    
    
if __name__ == "__main__":
    base_url = 'https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/Eurasia/'
    lis = zip_urls(base_url)
    get_zip(base_url,lis)
    
    


























