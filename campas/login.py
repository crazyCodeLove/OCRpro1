import urllib
import requests

def check(id, passwd):
    url = ""
    form_data = {'userId':id,
                 'password':passwd,
                 'passwordEncrypt':False}
    result = requests.post(url, data=form_data)
    print(result)

def decode():
    id = 15721066
    passwd=123456
    check(id, passwd)

def test():
    decode()

if __name__ == "__main__":
    test()
