# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

from firefin.data.gateway import fetch_data


data = fetch_data(["open", "test_no_data"])

print(data["open"])
print(data["test_no_data"])
