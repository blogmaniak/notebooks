# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:55:04 2018

@author: marjan
"""

from datetime import datetime

import time



## convert a time stamp to a date and time
dt = datetime.fromtimestamp(1346236702)

print dt

## convert a date and time to time stamp
timestamps = time.mktime(dt.timetuple())


print(dt.timetuple())
print timestamps