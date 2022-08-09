import hashlib
import random
import requests
import json
import time

def generateSalt():
    return str(random.randint(32768, 65536))

def getSign(appid, q, salt, secret_key, encoding='utf-8'):
    text = appid + q + salt + secret_key
    sign = hashlib.md5(text.encode(encoding))
    return sign.hexdigest()

class baiduTranslateAPI:
    def __init__(self):
        self.appid = '20220715001274106'
        self.secret_key = 'QVFFrQokZyRGF1pXzGm1'
        self.url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
        self.headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    def translate(self, q, src_lang='zh', dst_lang='en'):
        salt = generateSalt()
        sign = getSign(self.appid, q, salt, self.secret_key)
        data = {
            'q': q,
            'from': src_lang,
            'to': dst_lang,
            'appid': self.appid,
            'salt': salt,
            'sign': sign
        }
        retry = 0
        while True:
            try:  
                r = requests.post(self.url, data=data, headers=self.headers, timeout=15)
                break
            except Exception as e:
                retry += 1
                print('connect error, retry: %d' %retry)
            if retry > 2:
                break
            time.sleep(0.5)  
        if retry > 3:
            return -1, '[connect error]'
        try:
            res = json.loads(r.text)
            if 'trans_result' in res:
                return 0, res['trans_result'][0]['dst']
            else:
                return -1, res['error_msg']
        except Exception as e:
            return -1, 'json decode error'


if __name__ == '__main__':
    trans = baiduTranslateAPI()
    q = 'Lin Xi not only publicly responded positively to the Hong Kong independence leader, but even said in a speech at the University of Hong Kong, \"filling in the words for\" welcome to Beijing \"is an official mouthpiece and a\" stain on his life \".'
    code, res_data = trans.translate(q=q, src_lang='en', dst_lang='zh')
    print(code, res_data)