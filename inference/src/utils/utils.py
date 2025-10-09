import logging
logger = logging.getLogger(__name__)

class Utils:
    def timeit(func):
        def inner(*args,**kwargs):
            import datetime
            start_time = datetime.datetime.now()
            res = func(*args,**kwargs)
            end_time = datetime.datetime.now()
            logger.info(f"TIME TAKEN = {(end_time-start_time).total_seconds():.1f}sec")
            return res
        return inner