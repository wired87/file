import os
import shutil


def rm_tmp(tmp_store):
    if os.path.exists(tmp_store.name):
        shutil.rmtree(tmp_store.name)
    print("tmp store deleted")