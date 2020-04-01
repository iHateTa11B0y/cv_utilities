import requests
import cv2
from io import BytesIO
from PIL import Image
from functools import wraps
import numpy as np

# To speed up downloading
cv_session = requests.Session()
cv_session.trust_env = False

def retry(func):
    @wraps(func)
    def wrapper(*args, **kw):
        times = 4
        while times >= 0:
            try:
                return func(*args, **kw)
            except Exception as e:
                times -= 1
                print('Retry.')
        print('Failed.')
    return wrapper


class ImageLoader(object):
    """
       Initialize Arguments:
           core: specify image loader, use pillow or opencv
           channel_order: specify channel order 'rgb' or 'bgr'

       Functions:
           .load(inp, type_):
           Input:
               in_p: input
               type_: specify input is path or url
           Return:
               img: pillow.Image or numpy image array
    """
    def __init__(self, core='opencv', channel_order='RGB'):
        core = core.upper()
        channel_order = channel_order.upper()
        assert core in ('OPENCV', 'PIL'), 'Only support OPENCV, PIL for now.'
        assert channel_order in ('RGB', 'BGR'), 'Only support RGB, BGR order for now.'
        self.core = core
        self.channel_order = channel_order

    @retry    
    def load(self, in_p, type_='path'):
        if type_ == 'url' or in_p.startswith('http'):
            img = self._load_url(in_p)
        elif type_ == 'path':
            img = self._load_path(in_p)
        else:
            raise ValueError('invaild option for argument "type_". ')
        img = self._convert(img, self.channel_order)
        return img

    def _convert(self, img, channel_order=None):
        channel_order = self.channel_order if channel_order is None else channel_order
        assert channel_order in ('RGB', 'BGR'), 'Only support RGB, BGR order for now.'
        if self.core == 'PIL' and channel_order == 'BGR':
            r,g,b = img.split()
            img = Image.merge("RGB",(b,g,r))
        elif self.core == 'OPENCV' and channel_order == 'RGB':
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        return img 
        
    def _load_url(self, in_p):
        if self.core == 'PIL':
            return self._load_url_pil(in_p)
        else:
            return self._load_url_opencv(in_p)

    def _load_path(self, in_p):
        if self.core == 'PIL':
            return self._load_path_pil(in_p)
        else:
            return self._load_path_opencv(in_p)

    def _load_url_opencv(self, in_p):
        img_nparr = np.frombuffer(cv_session.get(in_p).content, np.uint8)
        img = cv2.imdecode(img_nparr, cv2.IMREAD_COLOR)
        return img

    def _load_path_opencv(self, in_p):
        img = cv2.imread(in_p, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        return img

    def _load_url_pil(self, in_p):
        try:
            imfile = requests.get(in_p)
            img = Image.open(BytesIO(imfile.content))
            return img.convert('RGB')
        except:
            return None

    def _load_path_pil(self, in_p):
        img = Image.open(in_p)
        return img.convert('RGB')
        

