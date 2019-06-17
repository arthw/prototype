import pickle
import logging
import datetime

lg = logging.getLogger(__name__)
lg.setLevel(level = logging.DEBUG)
console = logging.StreamHandler()
formatter = logging.Formatter('%(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)
lg.addHandler(console)


def save_pkl(filename, data):
    lg.debug("Save to %s, data size %d" % (filename, len(data[0])))
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(filename):
    data = None
    lg.debug("Load %s" % filename)
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    lg.debug("Data size %d from %s" % (len(data[0]), filename))
    return data

def time_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def stamp2date(stamp):
    return datetime.datetime.utcfromtimestamp(stamp).strftime('%Y-%m-%d')
