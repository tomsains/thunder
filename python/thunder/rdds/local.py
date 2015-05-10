from numpy import add, sum, ndarray, asarray


class LocalRDD(object):
    """
    A local collection of records, either key-value pairs
    or values only, that mimic the operations supported by
    Spark's RDD (resiliant distributed dataset)

    Parameters
    ----------
    values : list, ndarray, tuple
        The values of the RDD
    keys : list, ndarray, tuple, optional, default = None
        The keys of the RDD
    """

    def __init__(self, values, keys=None):
        if keys is None:
            self._records = self._tolist(values)
        else:
            self._records = zip(self._tolist(keys), self._tolist(values))

    @staticmethod
    def _tolist(records):
        if isinstance(records, ndarray):
            records = list(records)
        elif isinstance(records, tuple):
            records = list(records)
        elif not isinstance(records, list):
            raise TypeError("Records must be ndarray, tuple, or list, got %s" % type(records))
        return records

    def first(self):
        return self._records[0]

    def collect(self):
        return self._records

    def map(self, func):
        newvals = [func(x) for x in self._records]
        return LocalRDD(values=newvals)

    def mapValues(self, func):
        return self.map(lambda (k, v): (k, func(v)))

    def values(self):
        return self.map(lambda (k, v): v)

    def keys(self):
        return self.map(lambda (k, v): k)

    def reduce(self, func):
        return reduce(func, self._records)

    def filter(self, func):
        newvals = [x for x in self._records if func(x)]
        return LocalRDD(values=newvals)

    def foreach(self, func):
        for x in self._records:
            func(x)

    def groupByKey(self):
        newkeys = set(self.keys().collect())
        newvals = []
        for kk in newkeys:
            tmp = [v for (k, v) in self._records if k == kk]
            newvals.append(tmp)
        return LocalRDD(keys=list(newkeys), values=newvals)

    def reduceByKey(self, func):
        grouped = self.groupByKey()
        redfunc = lambda x: reduce(func, x)
        return grouped.mapValues(redfunc)

    def count(self):
        return len(self._records)

    def sum(self):
        return self.reduce(add)
