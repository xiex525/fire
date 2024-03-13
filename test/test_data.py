from navi_quant.data.gateway import fetch_data


data = fetch_data(['open','test_no_data'])

print(data['open'])
print(data['test_no_data'])