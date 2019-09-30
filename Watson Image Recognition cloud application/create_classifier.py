# Create IIW Image Recognition Application Python SDK, Watson Developer Cloud, Wastion Visual Recognition API 
import json
import os
from os.path import join, dirname
from os import environ
from watson_developer_cloud import VisualRecognitionV3

visual_recognition = VisualRecognitionV3(VisualRecognitionV3.latest_version, api_key='ebea3d6c70451937afbf1a5c8dd176b5b2b613cb')

with open(join(dirname(__file__), 'unmarkedjunctions.zip'), 'rb') as unmarked, \
    open(join(dirname(__file__), 'markedjunctions.zip'), 'rb') as marked, \
    open(join(dirname(__file__), 'boxjunctions.zip'), 'rb') as box:
 print ("Image files loading")
 print (json.dumps(visual_recognition.create_classifier('Junctions', \
   unmarked_positive_examples=unmarked, \
   marked_positive_examples=marked, \
   box_positive_examples=box), indent = 2))

