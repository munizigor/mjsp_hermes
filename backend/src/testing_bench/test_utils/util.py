import numpy as np
import time

# "%d-%m-%Y_%H:%M:%S"
def datetime_to_int(datetimes, f="%Y-%m-%d %H:%M:%S"):
    ints = np.array(
        [
            time.mktime(time.strptime(s, f)) if s is not None else np.nan
            for s in datetimes
        ]
    )
    print(ints)
    return ints





def get_hostname() -> str:
    import socket

    return socket.gethostname()