import os
import boto
import hashlib

access_key = os.environ["AWS_ACCESS_KEY_ID"]
secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]


def sync_data_with_s3(path, bucket):
    '''
    Purpose of this funcion is to update the data files with those saved on
    S3.  Files will only be downloaded if they have changed or are not
    contained in the local path
    '''
    # First ensure that the path exists and make it if not
    if not os.path.exists(path):
        os.makedirs(path)

    # Create a list of all files that are present in path including those
    # present in subdirectories
    files = []
    for dirname, dirnames, filenames in os.walk(path):
        files.extend([os.path.join(dirname, f) for f in filenames])
    files = [f.replace(path, '', 1)[1:] for f in files]

    # Make dictionary of local files with file name as the key and a list
    # containing bool value indicating whether the file is on S3 (set to False
    # initially) and the md5 hash of the file
    local_fdict = {f: [False, md5(os.path.join(path, f))] for f in files}

    # Create connection to s3 and connect to the bucket
    conn = boto.connect_s3(access_key, secret_access_key)
    b = conn.get_bucket(bucket)

    # Loop through every key in the bucket
    for key in b.list():
        # Make sure subfolders exist, if necessary
        copy_filestructure(path, key)

        # Set flag to True for file being in S3
        if key.name in local_fdict:
            local_fdict[key.name][0] = True

        # Determine whether the file should be downloaded
        download = check_file(key, local_fdict)

        # Delete file if it's present, but has changed
        if download and key.name in local_fdict:
            os.remove(os.path.join(path, key.name))

        if download:
            download_file(path, key)
        else:
            print 'File {} unchanged, skipping...'.format(key.name)

    # Remove files from local if not present on S3
    for k, v in local_fdict.iteritems():
        if not v[0]:
            print 'Removing {}...'.format(k)
            os.remove(os.path.join(path, k))
    print 'Sync Complete'


def download_file(path, key):
    print 'Downloading {} ({:.1f} MB)...'.format(key.name, key.size*float(1e-6))
    try:
        key.get_contents_to_filename(os.path.join(path, key.name))
    except:
        print 'ERROR: Failed to download {}'.format(key.name)


def check_file(key, local_fdict):
    ''' Return bool value indicating whether the file should be downloaded
    Reasons to download include if the file is present on your local but the file has changed on the S3 bucket (evaluated using the md5 hash) or if the file is not present on the local but is on S3.
    Return False if the file matches the file already present on S3
    '''
    md5_hash = local_fdict.get(key.name, (None, None))[1]
    return not md5_hash == key.etag[1:-1]


def md5(fname):
    ''' Calculate the md5 hash of local file '''
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def copy_filestructure(path, key):
    ''' Create subdirectories for key, if necessary '''
    # Determine path seperator which could be different on Windows
    sep = os.path.join(path, '')[-1]
    sub_path = sep.join(key.name.split(sep)[:-1])
    if not os.path.exists(os.path.join(path, sub_path)):
        os.makedirs(os.path.join(path, sub_path))


if __name__=='__main__':
    # Indicate where the contents of the S3 Bucket should go
    path = './data'
    # Indicate the name of the bucket to look to
    bucket = 'ewellingertesttest'

    sync_data_with_s3(path, bucket)
