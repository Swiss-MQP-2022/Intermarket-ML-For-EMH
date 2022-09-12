def pct_to_cumulative(data, initial):
    return (data + 1).cumprod() * initial
