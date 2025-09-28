from google.cloud import storage
from google.genai import types
import httpx


def get_image_part(image_url: str):
    # Step 3: Define the bucket and file path
    if image_url.startswith("gs://"):
        gcs_path = image_url
    else:
        gcs_path = image_url.replace("https://storage.googleapis.com/", "gs://")

    bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)

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


def generate_image_parts(image_url: str, caption: str = None):
    """Generates a list of parts for an image with an optional caption.

    Args:
        image_url: The URL of the image.
        caption: An optional caption for the image.

    Returns:
        A list of parts containing the image and caption.
    """
    parts = []
    if image_url is None:
        raise ValueError("Image URL is required when data_type is 'image'")
    if image_url.startswith("gs://"):
        # parts.append(types.Part.from_uri(image_url, mime_type="image/jpeg")) #TODO: Change in future
        parts.append(get_image_part(image_url))
    else:
        image = httpx.get(image_url)
        file_content = image.content
        parts.append(types.Part.from_bytes(data=file_content, mime_type="image/jpeg"))
    if caption:
        parts.append(
            types.Part.from_text(
                f"User sent in the above image with this caption: {caption}"
            )
        )
    else:
        parts.append(types.Part.from_text("User sent in the above image"))
    return parts


def generate_screenshot_parts(image_url: str, url: str = None):
    """Generates a list of parts for an image with an optional caption.

    Args:
        image_url: The URL of the image.
        caption: An optional caption for the image.

    Returns:
        A list of parts containing the image and caption.
    """
    parts = []
    if image_url is None:
        raise ValueError("Image URL is required when data_type is 'image'")
    if image_url.startswith("gs://"):
        # parts.append(types.Part.from_uri(image_url, mime_type="image/jpeg")) #TODO: Change in future
        parts.append(get_image_part(image_url))
    else:
        image = httpx.get(image_url)
        file_content = image.content
        parts.append(types.Part.from_bytes(data=file_content, mime_type="image/jpeg"))

    parts.append(types.Part.from_text(f"Screenshot of {url} above"))
    return parts


def generate_text_parts(text: str):
    """Generates a list of parts for a text input.

    Args:
        text: The text content to be added to the parts.

    Returns:
        A list of parts containing the text input.
    """
    if text is None:
        raise ValueError("Text content is required when data_type is 'text'")
    parts = [types.Part.from_text(f"User sent in: {text}")]
    return parts
