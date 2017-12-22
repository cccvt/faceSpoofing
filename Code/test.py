#!/usr/bin/python
import os, sys, subprocess, shlex, re
from subprocess import call
def probe_file(filename):
    cmnd = ['ffprobe', '-show_format', '-pretty', '-loglevel', 'quiet', filename]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print (filename)
    out, err =  p.communicate()
    print ("==========output==========")
    print (out)
    if err:
        print ("========= error ========")
        print (err)

# probe_file('../../faceSpoofing_Data/train/real/client001_session01_webcam_authenticate_adverse_1.mov')
# print( tl.getVideoDetails() )
metadata = os.stat('../../faceSpoofing_Data/train/real/client001_session01_webcam_authenticate_adverse_1.mov')
print("\n\n----METADATA----\n".format(metadata))