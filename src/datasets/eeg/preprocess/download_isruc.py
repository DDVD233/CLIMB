import requests


def download_isruc():
    base_url = 'http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupI-Extractedchannels'
    for subject in range(1, 101):
        filename = 'subject' + str(subject) + '.mat'
        url = base_url + '/' + filename
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)


if __name__ == '__main__':
    download_isruc()