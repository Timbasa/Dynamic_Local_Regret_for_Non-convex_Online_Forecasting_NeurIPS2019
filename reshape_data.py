import numpy as np
from datetime import datetime, timedelta


# transform the original datasets to the specific format
def reshape_data(data, time_flag):
    """
    base_dates[0] is the date of 2005-1-1 1:00, the start date of train model
    base_dates[1] to basedates[15](total 15 dates) are the start dates of L1-benchmark.csv to L15-benchmark.csv,
    or the start dates of L2-train.csv to L15-train.csv and solution15_L.csv
    """
    base_dates = [datetime(2005, 1, 1, 1, 00),
                  datetime(2010, 10, 1, 1, 00), datetime(2010, 11, 1, 1, 00), datetime(2010, 12, 1, 1, 00),
                  datetime(2011, 1, 1, 1, 00), datetime(2011, 2, 1, 1, 00), datetime(2011, 3, 1, 1, 00),
                  datetime(2011, 4, 1, 1, 00), datetime(2011, 5, 1, 1, 00), datetime(2011, 6, 1, 1, 00),
                  datetime(2011, 7, 1, 1, 00), datetime(2011, 8, 1, 1, 00), datetime(2011, 9, 1, 1, 00),
                  datetime(2011, 10, 1, 1, 00), datetime(2011, 11, 1, 1, 00), datetime(2011, 12, 1, 1, 00)]
    base_date = base_dates[time_flag]
    trend = np.arange(0, len(data), dtype=np.intc)
    date = list(map(lambda x: base_date + timedelta(hours=np.asscalar(x)), trend))
    weekday = list(map(lambda x: x.weekday(), date))
    month = list(map(lambda x: x.month, date))
    hour = list(map(lambda x: x.hour, date))

    day_of_week = np.zeros((len(data), 7))
    for i, w in enumerate(weekday):
        day_of_week[i, w] = 1

    day_of_month = np.zeros((len(data), 12))
    for i, w in enumerate(month):
        day_of_month[i, w - 1] = 1

    day_of_hour = np.zeros((len(data), 24))
    for i, w in enumerate(hour):
        day_of_hour[i, w - 1] = 1

    reshaped_data = np.zeros((data.shape[0], 44))

    for i in range(reshaped_data.shape[0]):
        reshaped_data[i][0] = data[i][0]
        reshaped_data[i][1:25] = day_of_hour[i]
        reshaped_data[i][25:32] = day_of_week[i]
        reshaped_data[i][32:44] = day_of_month[i]

    return reshaped_data
