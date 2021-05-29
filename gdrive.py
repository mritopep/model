import pickle
import os
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from paths import *
from general import store_data
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from mimetypes import MimeTypes
from googleapiclient.errors import HttpError
import io


class Gdrive:
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    def __init__(self):
        self.creds = self.load_cred('token.pickle')
        if(self.creds == None):
            self.creds = self.login(self.creds)
        self.service = build('drive', 'v3', credentials=self.creds)
        

    def login(self,creds):
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    f'{TOKEN}/credentials.json', self.SCOPES)
                creds = flow.run_local_server(port=0)
            with open(f'{TOKEN}/token.pickle', 'wb') as token:
                pickle.dump(creds, token)
            return creds


    def load_cred(self,name):
        if os.path.exists(f'{TOKEN}/{name}'):
            with open(f'{TOKEN}/{name}', 'rb') as token:
                creds = pickle.load(token)
                return creds
        else:
            return None


    def get_file_id(self):
        files = []
        items = []
        page_token = None
        while True:
            response = self.service.files().list(fields='nextPageToken, files(id, name)',
                                            pageToken=page_token).execute()
            items += response.get('files', [])
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
        if items:
            for item in items:
                files.append(item)
        else:
            print("-")
        return files


    def upload(self, path, parent_id=None):
        mime = MimeTypes()
        file_metadata = {
            'name': os.path.basename(path),
        }

        if parent_id:
            file_metadata['parents'] = [parent_id]

        media = MediaFileUpload(path,
                                mimetype=mime.guess_type(
                                    os.path.basename(path))[0],
                                resumable=True)
        try:
            file = self.service.files().create(body=file_metadata,
                                          media_body=media,
                                          fields='id').execute()
            print('File ID: %s' % file.get('id'))
        except HttpError:
            print('corrupted file')
            pass
        


    def share(self, file_id, email):
        def callback(request_id, response, exception):
            if exception:
                print(exception)
            else:
                print(response.get('id'))

        batch = self.service.new_batch_http_request(callback=callback)
        user_permission = {
            'type': 'user',
            'role': 'reader',
            'emailAddress': email
        }
        batch.add(self.service.permissions().create(
            fileId=file_id,
            body=user_permission,
            fields='id',
        ))
        batch.execute()


    def delete(self, fileid):
        self.service.files().delete(fileId=fileid).execute()

    def download(self,file_id, path=os.getcwd()):
        request = self.service.files().get_media(fileId=file_id)
        name = self.service.files().get(fileId=file_id).execute()['name']
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(int(status.progress() * 100))
        f = open(path + '/' + name, 'wb')
        f.write(fh.getvalue())
        print('File downloaded at', path)
        f.close()

    def createfolder(self,folder, recursive=False):
        if recursive:
            print('recursive ON')
            ids = {}
            for root, sub, files in os.walk(folder):
                par = os.path.dirname(root)

                file_metadata = {
                    'name': os.path.basename(root),
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                if par in ids.keys():
                    file_metadata['parents'] = [ids[par]]
                print(root)
                file = self.service.files().create(body=file_metadata,
                                              fields='id').execute()
                id = file.get('id')
                print('File ID: %s' % file.get('id'))
                ids[root] = id
                for f in files:
                    print(root+'/'+f)
                    upload(root + '/' + f, id)
        else:
            print('recursive OFF')
            file_metadata = {
                    'name': os.path.basename(folder),
                    'mimeType': 'application/vnd.google-apps.folder'
                }
            file = self.service.files().create(body=file_metadata,
                                              fields='id').execute()
            print('File ID: %s' % file.get('id'))

if __name__ == '__main__':
    g = Gdrive()
    print(g.get_file_id())