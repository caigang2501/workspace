import re

m = re.match(r"\w+\s+\w+", "Malcolm   Reynolds")
print((m.groups(1)))