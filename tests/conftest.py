# -*- coding: utf-8 -*-
"""pytest 公共配置：把项目根加入 sys.path，便于 import backend / scripts"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
