import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import torch


# Replace with your file ID
# file_id = '1HeokPU-JNVJuPOQwcFwzk3if-NwnuLGs'
# file_id = "1OS1nEZ5IUYkwAkSK4lWL8b27a4G3YkPK"

def get_model_from_drive(file_id, credentials):

    # Build the Drive service
    service = build('drive', 'v3', credentials=credentials)

    # Download the file
    request = service.files().get_media(fileId=file_id)
    file_content = io.BytesIO()
    downloader = MediaIoBaseDownload(file_content, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    file_content.seek(0)  # rewind buffer to start
    
    return file_content
    # Now load the .pth file into a PyTorch model or state_dict
    # state_dict = torch.load(file_content, map_location=torch.device('cpu'))