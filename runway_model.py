import os
import face_recognition
import numpy as np
import runway
from runway.data_types import category, image, vector, array, any
from PIL import Image

# in: (top, right, bottom, left)
# out: (left, top, right, bottom)
def fr_rect_to_pil_rect(rect):
    return rect[3], rect[0], rect[1], rect[2]

# in: (left, top, right, bottom)
# out: (left, top, width, height)
def pil_rect_to_x_y_w_h(pil_rect):
    x = pil_rect[0]
    y = pil_rect[1]
    w = pil_rect[2] - x
    h = pil_rect[3] - y
    return x, y, w, h

# https://github.com/ageitgey/face_recognition/blob/c96b010c02f15e8eeb0f71308c641179ac1f19bb/examples/facerec_from_webcam_faster.py#L60
identify_face_inputs = { 'input_image': image, 'label_image': image }
identify_face_outputs = {
    'results': array(item_type=any),
    'size': any
}
@runway.command('Identify_Face', inputs=identify_face_inputs, outputs=identify_face_outputs)
def identify_face(model, args):
    input_arr = np.array(args['input_image'])
    label_arr = np.array(args['label_image'])
    input_encodings = face_recognition.face_encodings(input_arr)
    label_encodings = face_recognition.face_encodings(label_arr)

    width = input_arr.shape[1]
    height = input_arr.shape[0]
    results = []
    if len(input_encodings) > 0 and len(label_encodings) > 0:
        # compare the labeled encoding to each face found in the input image
        matches = face_recognition.compare_faces(input_encodings, label_encodings[0])
        if True in matches:
            faces = face_recognition.face_locations(input_arr)
            faces = [ pil_rect_to_x_y_w_h(fr_rect_to_pil_rect(face)) for face in faces ]
            match_index = matches.index(True)
            results = [{ 'bbox': faces[match_index], 'class': 'Match Found' }]
    return { 'results': results, 'size': { 'width': width, 'height': height } }

detect_faces_output = {
    'results': array(item_type=any),
    'size': any
}
@runway.command('Detect_Faces', inputs={ 'image': image }, outputs=detect_faces_output)
def detect_faces(model, args):

    np_arr = np.array(args['image'])
    width = np_arr.shape[1]
    height = np_arr.shape[0]
    faces = face_recognition.face_locations(np_arr)

    results = []
    faces = [ pil_rect_to_x_y_w_h(fr_rect_to_pil_rect(face)) for face in faces ]
    for i in range(len(faces)):
        results.append({ 'bbox': faces[i], 'class': 'Face #{}'.format(i + 1)})
    return { 'results': results, 'size': { 'width': width, 'height': height } }

if __name__ == '__main__':
    runway.run()
