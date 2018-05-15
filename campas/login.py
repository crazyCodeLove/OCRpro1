import urllib
import requests

def check(id, passwd):
    url = "http://10.10.9.9:8080/eportal/index.jsp?wlanuserip=10.88.53.122&wlanacname=M18014-WS&ssid=&nasip=202.120.127.125&snmpagentip=&mac=9cb6d01a38bd&t=wireless-v2-plain&url=http://www.msftconnecttest.com/redirect&apmac=&nasid=M18014-WS&vid=947&port=50&nasportid=AggregatePort%204.09470000:947-0"
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
