#coding=utf-8

import smtplib
from email.mime.text import MIMEText


class Email(object):

    def __init__(self,username,passwd):
        self._mail_host = "smtp.163.com" # 设置服务器
        self._mail_user = username  # 用户名
        self._mail_pass = passwd


    def send_mail_html(self,to_list, sub, content):
        from_addr = self._mail_user +"@163.com"
        msg = MIMEText(content, _subtype='html', _charset='utf-8')  # 创建一个实例，这里设置为html格式邮件
        msg['Subject'] = sub  # 设置主题
        msg['From'] = from_addr
        msg['To'] = ";".join(to_list)
        try:
            s = smtplib.SMTP()
            s.connect(self._mail_host)  # 连接smtp服务器
            s.login(self._mail_user, self._mail_pass)  # 登陆服务器
            s.sendmail(from_addr, to_list, msg.as_string())  # 发送邮件
            s.close()
            return True
        except Exception as e:
            print(str(e))
            return False






if __name__ == '__main__':
    username = "xxxxx"
    passwd = "xxxxx"
    email = Email(username,passwd)

    mailto_list = ["xxx@163.com"]
    title = "test"
    content = "<a href='http://www.cnblogs.com/xiaowuyi'>小五义</a>"

    if email.send_mail_html(mailto_list,title,content):
        print("发送成功")
    else:
        print("发送失败")
