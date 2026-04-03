# -*- coding: utf-8 -*-'
#!/usr/bin/env python3
import re
import pickle
import time
from datetime import datetime
import jieba  
import warnings


warnings.filterwarnings("ignore")
jieba.load_userdict("config/user_dic_expand.txt")

sentence="双肺纹理增粗"
print(jieba.lcut(sentence))