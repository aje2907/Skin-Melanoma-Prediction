from google.auth import credentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.credentials = credentials.Credentials.from_authorized_user_info(info=None)

drive = GoogleDrive(gauth)
