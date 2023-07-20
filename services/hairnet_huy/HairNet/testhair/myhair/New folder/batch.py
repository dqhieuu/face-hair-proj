import requests


def download(fileName):
    f = open(fileName,'wb')
    f.write(requests.get('https://thispersondoesnotexist.com', headers={'User-Agent': 'My User Agent 1.0'}).content)
    f.close()

for i in range(2000):
    download(str(i)+'.jpg')