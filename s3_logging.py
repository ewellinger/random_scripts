from __future__ import absolute_import, division, print_function
from builtins import *
import boto3
import botocore
from io import StringIO
from datetime import datetime
import sys


class S3Logging(object):
    """
    Object allowing for printing logs to an S3 file

    Useful for iteratively logging messsages in a long running script such as a
    Spark application where stdout is only available upon completion.

    NOTE: S3 does not allow appending to an already existing file and so your
    specified log will be rewritten upon each call to `push_log()`

    NOTE: Must have previously configured the awscli tools
    """
    def __init__(self, bucket, fname, tstamp=True, redirect_stderr=False, push=False, overwrite_existing=False):
        """
        Args:
            bucket (str): S3 Bucket name
            fname (str): Name to give to log file
            tstamp (bool): default True
                Whether to include a timestamp with each call to write
            redirect_stderr (bool): default False
                Direct all stderr messages to be logged
            push (bool): default False
                Copy log to S3 upon each call to write()
            overwrite_existing (bool): default False
                Whether to overwrite file if it already exists.  If False and
                the file does already exist, messages will be appended to the
                file
        """
        self._s3 = boto3.client('s3')
        self.bucket = bucket
        self.key = fname
        self._tstamp = tstamp
        self._push = push

        if redirect_stderr:
            # redirect all stderr outputs to write to self
            sys.stderr = self

        if not overwrite_existing and self._exists():
            body_obj = self._s3.get_object(Bucket=self.bucket, Key=self.key)['Body']
            self._msg = str(body_obj.read(), 'utf-8')
        else:
            self._msg = ''

    def write(self, msg, push=None):
        if push is None:
            push = self._push

        # Append message with or without timestamp
        if self._tstamp:
            self._msg += "\n{0}\n{1}\n".format(datetime.now(), msg)
        else:
            self._msg += "\n{0}\n".format(msg)

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

    log = S3Logging(bucket, fname, overwrite_existing=True)
    log.write("This is a test1")
    print("This should still be printed to the normal stderr", file=sys.stderr)
    log.push_log()

    log = S3Logging(bucket, fname, redirect_stderr=True)
    log.write("This is a second test!")
    print("This should be redirected into the log object!", file=sys.stderr)
    print(log)
