from future.builtins import *
import boto3
import botocore
from io import StringIO
from datetime import datetime


class S3Logging(object):
    """
    Object allowing for printing logs to an S3 file

    Useful for iteratively logging messsages in a long running script such as a
    Spark application where stdout is only available upon completion.
    NOTE: S3 does not allow appending to an already existing file and so your
    specified log will be rewritten upon each call to `push_log()`
    NOTE: Must have previously configured the awscli tools
    """
    def __init__(self, bucket, fname):
        """
        Args:
            bucket (str): S3 Bucket name
            fname (str): Name to give to log file
        """
        self._s3 = boto3.client('s3')
        self.bucket = bucket
        self.key = fname

        if self._exists():
            body_obj = self._s3.get_object(Bucket=self.bucket, Key=self.key)['Body']
            self._msg = str(body_obj.read(), 'utf-8')
        else:
            self._msg = ''

    def log(self, msg, push=False):
        self._msg += "\n{0}\n{1}\n".format(datetime.now(), msg)
        if push:
            self.push_log()

    def push_log(self):
        f_handle = StringIO(self._msg)
        self._s3.put_object(Bucket=self.bucket, Key=fname, Body=f_handle.read())

    def _exists(self):
        bucket = boto3.resource('s3').Bucket(self.bucket)
        objs = list(bucket.objects.filter(Prefix=self.key))
        return len(objs) > 0 and objs[0].key == self.key

    def __repr__(self):
        return self._msg



if __name__=='__main__':
    bucket = 'ewellingertesttest'
    fname = 'test_log.txt'

    log = S3Logging(bucket, fname)

    log.log("This is a test")

    log.log({1: 'a', 2: 'b'}, True)
