from google.cloud import storage
from google.genai import types


def get_image_part(gcs_path):
    # Step 3: Define the bucket and file path
    bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
    #
    # Initialize the client#
    client = storage.Client()  #
    #
    # Get the bucket#
    bucket = client.bucket(bucket_name)  #
    #
    # Get the blob (file)#
    blob = bucket.blob(blob_name)

    # Download the blob content as bytes
    file_content = blob.download_as_bytes()

    # # Step 4: Encode the file content as Base64
    # base64_encoded_string = base64.b64encode(file_content).decode('utf-8')

    return types.Part.from_bytes(data=file_content, mime_type="image/jpeg")
