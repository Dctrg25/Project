
import time
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
output= {'Unnamed: 0': [], 'Giới tính': [], 'Hình thức làm việc': [], 'Kinh nghiệm': [], 'Mức lương': [], 'Ngành nghề': [], 'Số lượng tuyển dụng': [], 'Thời gian thử việc': [], 'Trình độ': [], 'Tính chất công việc': [], 'Tỉnh/Thành phố': [], 'benefits': [], 'description': [], 'job_title': [], 'link': [], 'requirements': []} 
browser=webdriver.Edge()
url='https://vieclam24h.vn/mien-nam/viec-lam-ban-buon-ban-le-quan-ly-cua-hang-o6.html'
browser.get(url)
time.sleep(20)
def Crawl_Job():
    for x in range(1,21):
        try:
            title_job=browser.find_element(By.XPATH, "/html/body/div[1]/main/div/div[3]/div[4]/div/div/div[1]/div[2]/table/tbody/tr["+str(x)+"]/td[1]/div/a[1]")
            title_job.click()
            time.sleep(2)

            muc_luong=browser.find_element(By.XPATH, '/html/body/div[1]/main/div/div[3]/div[5]/div/div/div[1]/div/article/div[3]/div[1]/ul/li[1]')
            kinh_nghiem=browser.find_element(By.XPATH, '/html/body/div[1]/main/div/div[3]/div[5]/div/div/div[1]/div/article/div[3]/div[1]/ul/li[2]')
            trinh_do=browser.find_element(By.XPATH, '/html/body/div[1]/main/div/div[3]/div[5]/div/div/div[1]/div/article/div[3]/div[1]/ul/li[3]')
            Tinh_thanhpho=browser.find_element(By.XPATH, '/html/body/div[1]/main/div/div[3]/div[5]/div/div/div[1]/div/article/div[3]/div[1]/ul/li[4]')
            nganh_nghe=browser.find_element(By.XPATH, '/html/body/div[1]/main/div/div[3]/div[5]/div/div/div[1]/div/article/div[3]/div[1]/ul/li[5]')
            so_luong=browser.find_element(By.XPATH,   '/html/body/div[1]/main/div/div[3]/div[5]/div/div/div[1]/div/article/div[3]/div[2]/ul/li[1]')
            gioi_tinh=browser.find_element(By.XPATH,   '/html/body/div[1]/main/div/div[3]/div[5]/div/div/div[1]/div/article/div[3]/div[2]/ul/li[2]')
            tinh_chat_cong_viec=browser.find_element(By.XPATH,   '/html/body/div[1]/main/div/div[3]/div[5]/div/div/div[1]/div/article/div[3]/div[2]/ul/li[3]')
            hinh_thuc_lam_viec=browser.find_element(By.XPATH,   '/html/body/div[1]/main/div/div[3]/div[5]/div/div/div[1]/div/article/div[3]/div[2]/ul/li[4]')
            thoi_gian_thu_viec=browser.find_element(By.XPATH,   '/html/body/div[1]/main/div/div[3]/div[5]/div/div/div[1]/div/article/div[3]/div[2]/ul/li[5]')
            description=browser.find_element(By.XPATH,'/html/body/div[1]/main/div/div[3]/div[5]/div/div/div[1]/div/article/table/tbody/tr[1]/td[2]')
            requirements=browser.find_element(By.XPATH,'/html/body/div[1]/main/div/div[3]/div[5]/div/div/div[1]/div/article/table/tbody/tr[2]/td[2]')
            benefits=browser.find_element(By.XPATH,'/html/body/div[1]/main/div/div[3]/div[5]/div/div/div[1]/div/article/table/tbody/tr[3]/td[2]')

            output['Mức lương'].append(str(muc_luong.text.replace("- Mức lương:", "")))
            output['Kinh nghiệm'].append(str(kinh_nghiem.text.replace("- Kinh nghiệm:", "")))
            output['Trình độ'].append(trinh_do.text.replace("- Trình độ:", ""))
            output['Tỉnh/Thành phố'].append(Tinh_thanhpho.text.replace("- Tỉnh/Thành phố:", ""))
            output['Ngành nghề'].append(nganh_nghe.text.replace("- Ngành nghề:", ""))
            output['Số lượng tuyển dụng'].append(so_luong.text.replace("- Số lượng tuyển dụng:", ""))
            output['Giới tính'].append(gioi_tinh.text.replace("- Giới tính:", ""))
            output['Tính chất công việc'].append(tinh_chat_cong_viec.text.replace("- Tính chất công việc:", ""))
            output['Hình thức làm việc'].append(hinh_thuc_lam_viec.text.replace("- Hình thức làm việc:", ""))
            output['Thời gian thử việc'].append(thoi_gian_thu_viec.text.replace("- Thời gian thử việc:", ""))
            output['description'].append(description.text.replace("Mô tả", ""))
            output['requirements'].append(requirements.text.replace("Yêu cầu", ""))
            output['benefits'].append(benefits.text.replace("Quyền lợi", ""))
            browser.back()
            time.sleep(2)
        except:
            continue
pagenumber = 1
while pagenumber < 6 :
    Crawl_Job()
    if pagenumber <3:
        next = browser.find_element(By.XPATH,'/html/body/div[1]/main/div/div[3]/div[4]/div/div/div[1]/div[2]/div[3]/a[7]')
        next.click()
        time.sleep(2)
    elif pagenumber == 3:
        next = browser.find_element(By.XPATH,'/html/body/div[1]/main/div/div[3]/div[4]/div/div/div[1]/div[2]/div[3]/a[8]')
        next.click()
        time.sleep(2)

    else :
        next = browser.find_element(By.XPATH,'/html/body/div[1]/main/div/div[3]/div[4]/div/div/div[1]/div[2]/div[3]/a[9]')
        next.click()
        time.sleep(2)
    pagenumber += 1
browser.quit()
df = pd.DataFrame.from_dict(output, orient='index') #key is rows
df = df.transpose() #transpose rows and columns
df.to_csv('JD_timviecnhanh_laodongphothong.csv', index=False)