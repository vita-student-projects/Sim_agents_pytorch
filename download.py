import os
import re
from tqdm import tqdm
from google.auth import default, exceptions
from google.cloud import storage

# Set the environment variable to use your Google credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'application_default_credentials.json'

def download_from_gcs(gcs_path):
    # Extract the filename from the file URL
    filename = re.search('/(training|testing|validation)/(.+\.tfrecord-\d+-of-\d+)', gcs_path).group(2)
    # see wether it is a training, testing or validation file
    file_type = re.search('/(training|testing|validation)/(.+\.tfrecord-\d+-of-\d+)', gcs_path).group(1)
    # If the waymo open dataset folder does not exist, create it
    if not os.path.exists('waymo_open_dataset_'):
        os.makedirs('waymo_open_dataset_')
    # If the training, testing or validation folder does not exist, create it
    for folder in ['training', 'testing', 'validation']:
        if not os.path.exists('waymo_open_dataset_/' + folder):
            os.makedirs('waymo_open_dataset_/' + folder)
    # Local path
    local_path = 'waymo_open_dataset_/' + file_type + '/' + filename
    # Check if the file already exists
    if os.path.isfile(local_path):
        print(f'{filename} already exists in {local_path}')
    else:
        try:
            # Get the default credentials using your Google credentials
            credentials, _ = default()
            # Create a storage client using the credentials
            client = storage.Client(credentials=credentials, project='223476880990')
            # Get a reference to the bucket and object
            bucket_name = 'waymo_open_dataset_motion_v_1_2_0'
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(gcs_path)

            # Download the object to a file with a progress bar
            with tqdm.wrapattr(open(local_path, 'wb'), 'write', miniters=1,
                               total=blob.size, desc=f'Downloading {gcs_path}') as file_obj:
                blob.download_to_file(file_obj)

            print(f'Object downloaded to {local_path}')
        except exceptions.DefaultCredentialsError:
            print('Unable to obtain default Google credentials. Make sure you have set up your credentials properly.')