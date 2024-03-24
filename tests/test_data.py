# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/auderson/FactorInvestmentResearchEngine/blob/master/NOTICE.txt

from fire.data.gateway import fetch_data


data = fetch_data(["open", "test_no_data"])

print(data["open"])
print(data["test_no_data"])
