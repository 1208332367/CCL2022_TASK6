import requests
import json
import random

ua_list = [
    "Mozilla/5.0 (iPod; U; CPU iPhone OS 4_3_2 like Mac OS X; zh-cn) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8H7 Safari/6533.18.5",
    "Mozilla/5.0 (iPhone; U; CPU iPhone OS 4_3_2 like Mac OS X; zh-cn) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8H7 Safari/6533.18.5",
    "MQQBrowser/25 (Linux; U; 2.3.3; zh-cn; HTC Desire S Build/GRI40;480*800)",
    "Mozilla/5.0 (Linux; U; Android 2.3.3; zh-cn; HTC_DesireS_S510e Build/GRI40) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
    "Mozilla/5.0 (SymbianOS/9.3; U; Series60/3.2 NokiaE75-1 /110.48.125 Profile/MIDP-2.1 Configuration/CLDC-1.1 ) AppleWebKit/413 (KHTML, like Gecko) Safari/413",
    "Mozilla/5.0 (iPad; U; CPU OS 4_3_3 like Mac OS X; zh-cn) AppleWebKit/533.17.9 (KHTML, like Gecko) Mobile/8J2",
    "Mozilla/5.0 (Windows NT 5.2) AppleWebKit/534.30 (KHTML, like Gecko) Chrome/12.0.742.122 Safari/534.30",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_2) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/14.0.835.202 Safari/535.1",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_2) AppleWebKit/534.51.22 (KHTML, like Gecko) Version/5.1.1 Safari/534.51.22",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 5_0 like Mac OS X) AppleWebKit/534.46 (KHTML, like Gecko) Version/5.1 Mobile/9A5313e Safari/7534.48.3",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 5_0 like Mac OS X) AppleWebKit/534.46 (KHTML, like Gecko) Version/5.1 Mobile/9A5313e Safari/7534.48.3",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 5_0 like Mac OS X) AppleWebKit/534.46 (KHTML, like Gecko) Version/5.1 Mobile/9A5313e Safari/7534.48.3",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/14.0.835.202 Safari/535.1",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows Phone OS 7.5; Trident/5.0; IEMobile/9.0; SAMSUNG; OMNIA7)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0; XBLWP7; ZuneWP7)",
    "Mozilla/5.0 (Windows NT 5.2) AppleWebKit/534.30 (KHTML, like Gecko) Chrome/12.0.742.122 Safari/534.30",
    "Mozilla/5.0 (Windows NT 5.1; rv:5.0) Gecko/20100101 Firefox/5.0",
    "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.2; Trident/4.0; .NET CLR 1.1.4322; .NET CLR 2.0.50727; .NET4.0E; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729; .NET4.0C)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; .NET4.0E; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729; .NET4.0C)",
    "Mozilla/4.0 (compatible; MSIE 60; Windows NT 5.1; SV1; .NET CLR 2.0.50727)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)",
    "Opera/9.80 (Windows NT 5.1; U; zh-cn) Presto/2.9.168 Version/11.50",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)",
    "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.04506.648; .NET CLR 3.5.21022; .NET4.0E; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729; .NET4.0C)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/533.21.1 (KHTML, like Gecko) Version/5.0.5 Safari/533.21.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; ) AppleWebKit/534.12 (KHTML, like Gecko) Maxthon/3.0 Safari/534.12",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 2.0.50727; TheWorld)"
]

class baiduTranslateCrawling:
    def __init__(self):
        self.url = 'http://fanyi.baidu.com/v2transapi'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36 Edg/103.0.1264.62',
            'cookie': 'BIDUPSID=6EA3FDF23AC25E759D2D2D989E3F7C7A; PSTM=1655962676; BAIDUID=6EA3FDF23AC25E75CA4750CAEDEF50FD:SL=0:NR=10:FG=1; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; APPGUIDE_10_0_2=1; FANYI_WORD_SWITCH=1; HISTORY_SWITCH=1; REALTIME_TRANS_SWITCH=1; SOUND_SPD_SWITCH=1; SOUND_PREFER_SWITCH=1; BDUSS=zJDWmpEcjBhZlJIa01LflF5NHE2ZG5-V1hFdDVIaWdndjlYV2p-eU1Xcko1UGhpSVFBQUFBJCQAAAAAAAAAAAEAAACT1Jy9YTEyMDgzMzIzNjcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMlX0WLJV9FiS; BDUSS_BFESS=zJDWmpEcjBhZlJIa01LflF5NHE2ZG5-V1hFdDVIaWdndjlYV2p-eU1Xcko1UGhpSVFBQUFBJCQAAAAAAAAAAAEAAACT1Jy9YTEyMDgzMzIzNjcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMlX0WLJV9FiS; Hm_lvt_64ecd82404c51e03dc91cb9e8c025574=1657370767,1657886236,1657967502; Hm_lpvt_64ecd82404c51e03dc91cb9e8c025574=1657974931; BA_HECTOR=0l0la10ga58g21a48g80bu831hd5d2g16; delPer=0; PSINO=3; ZFY=F1J0u5byXKSajgFoFZCtQQgO4Lbay:AfCDJGKvWyHxgc:C; BAIDUID_BFESS=6EA3FDF23AC25E75CA4750CAEDEF50FD:SL=0:NR=10:FG=1; ZD_ENTRY=baidu; ab_sr=1.0.1_ZGQwYWE2MTFiNTM3MmY5ZjFmMGM5MjBhOTc4NjMxYzA0YTM5ODNjNGI4MTljZjE1MDk4ZjA2NmEwNTVjM2Y2M2RlN2ZhOTg1Y2FiMWY2Yzk2ODM4ZWY5NDJkY2I1MGIwYzllMDViOTQ4NzU2YzcyNjI1OGIwMDg5ZDRiYzJiN2ZlMjdhNDMwNWViNzhlNTRkMTRiM2I4MjQ1NzY2ODVhNzQ0OTg3MzA1OWE0YzBlNTM3OTQwMzZiYjY1MDMwYTAx; H_PS_PSSID=36559_36464_36255_36825_36454_36414_34812_36692_36165_36816_36804_36776_36775_36636_36746_36760_36769_36765_26350_36687_36865'
        }

    def translate(self, q, src_lang='zh', dst_lang='en'):
        data = {
            'query': q,
            'from': src_lang,
            'to': dst_lang
        }
        r = requests.post(self.url, data=data, headers=self.headers)
        try:
            res = json.loads(r.text)
            print(res)
            if 'trans_result' in res:
                if res['trans_result']['status'] == 0:
                    return 0, res['trans_result']['data'][0]['dst']
            return -1, 'get translate result error'
        except Exception as e:
            return 500, 'json decode error'


if __name__ == '__main__':
    # example
    trans = baiduTranslateCrawling()
    q = 'Meng Wanzhou replied, \"Madam judge, I heard it on the phone.'
    code, res_data = trans.translate(q=q, src_lang='en', dst_lang='zh')
    print(code, res_data)