import io
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import torch


# Replace with your file ID
# file_id = '1HeokPU-JNVJuPOQwcFwzk3if-NwnuLGs'
# file_id = "1OS1nEZ5IUYkwAkSK4lWL8b27a4G3YkPK"

def get_model_from_drive(file_id):
    # Define the scope
    scope = ['https://www.googleapis.com/auth/drive']

    # Authenticate using the service account JSON key
    gauth = GoogleAuth()
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name('webapp-433015-3359b2e5fd44.json', scope)

    drive = GoogleDrive(gauth)

    # Step 3: Fetch the file from Google Drive
    service = build('drive', 'v3', credentials=gauth.credentials)

    request = service.files().get_media(fileId=file_id)

    # Create a BytesIO object to receive the file content
    file_content = io.BytesIO()

    # Download the file into the BytesIO object
    downloader = MediaIoBaseDownload(file_content, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()

    # Rewind the buffer to the beginning so it can be read by torch.load
    file_content.seek(0)

    return file_content

    # Now load the .pth file into a PyTorch model or state_dict
    # state_dict = torch.load(file_content, map_location=torch.device('cpu'))