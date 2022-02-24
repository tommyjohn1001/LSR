import pickle


def save_object(path, ob):
    with open(path, "wb") as handle:
        pickle.dump(ob, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)
