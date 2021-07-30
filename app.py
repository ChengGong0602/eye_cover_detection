from flask import Flask, jsonify, request
import cv2
import mediapipe as mp
import time
import numpy as np
import sys
import base64

app = Flask (__name__)
 
@app.route('/')
def hello_world():
    return 'Hello, World!'


def readb64(base64_string):
    # sbuf = StringIO()
    # sbuf.write(base64.b64decode(base64_string))
    # pimg = Image.open(sbuf)
    # return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)    
    imgString = base64_string.encode().split(b';base64,')[-1]  
    decoded_data = base64.b64decode(imgString)
    np_data = np.fromstring(decoded_data,np.uint8)
    return cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.minDetectionCon=minDetectionCon
        self.minTrackCon =minTrackCon

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
                                            self.staticMode, self.maxFaces,
                                            self.minDetectionCon, self.minTrackCon)
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    def findFaceMesh(self, img, draw = True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image.flags.writeable = False
        self.results = self.face_mesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:                    
                    self.mp_drawing.draw_landmarks(
                        image=self.imgRGB,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACE_CONNECTIONS,
                        landmark_drawing_spec=self.drawing_spec,
                        connection_drawing_spec=self.drawing_spec)
                face = []
                # get landmark 
                for id,lm in enumerate(face_landmarks.landmark):
                    # print(lm)
                    ih, iw, ic = self.imgRGB.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    face.append([x,y])
                    # if id in [ 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,       362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]:  
                    #     cv2.putText(img, str(id), (x,y),cv2.FONT_HERSHEY_PLAIN, 0.5, (0,255,0),1)
                # print(id,x,y)
                faces.append(face)
        return img, faces

def eye_coverage_estimate(image, faces, color_th):
    face = faces[0]
    mask = np.zeros(image.shape[0:2], dtype=np.uint8)
    Eye_cover="black"
    if Eye_cover=="white":
        lefteye_points = np.array([
            face[33], face[246], face[161], face[160], face[159], face[158], face[157], face[173], 
            face[133], face[155], face[154], face[153], face[145], face[144], face[163], face[7],
            ])
        righteye_points=np.array([
            face[362], face[398], face[384], face[385], face[386], face[387], face[388], face[466], 
            face[263],face[249], face[390], face[373], face[374], face[380], face[381], face[382], 
            ])
    else:
        lefteye_points = np.array([
            face[226], face[113], face[225], face[224], face[223], face[222], face[221], face[189], 
            face[244], face[233], face[232], face[231], face[230], face[229], face[228], face[31]
            ])
        righteye_points=np.array([
            face[464], face[413], face[441], face[442], face[443], face[444], face[445], face[342], 
            face[446],face[261], face[448], face[449], face[450], face[451], face[452], face[453], 
            ])
    
    #method 1 smooth region
    cv2.drawContours(mask, [lefteye_points.astype(int)], -1, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.drawContours(mask, [righteye_points.astype(int)], -1, (255, 255, 255), -1, cv2.LINE_AA)
    

    #method 2 not so smooth region
    # cv2.fillPoly(mask, [lefteye_points.astype(int)], (255))
    # cv2.fillPoly(mask, [righteye_points.astype(int)], (255))
    
    # Crop eye rect
    res = cv2.bitwise_and(image,image,mask = mask)
    # ## crate the white background of the same size of original image
    # wbg = np.ones_like(frame, np.uint8)*255
    # cv2.bitwise_not(wbg,wbg, mask=mask)
    # # overlap the resulted cropped image on the white background
    # res = wbg+res
    gray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    if Eye_cover=="white":
        ret,res=cv2.threshold(gray,color_th,255,cv2.THRESH_BINARY)
    else:
        ret,res=cv2.threshold(gray,color_th,255,cv2.THRESH_BINARY)
    
    lefteye_rect = cv2.boundingRect(lefteye_points) # returns (x,y,w,h) of the rect
    lefteye_cropped = res[lefteye_rect[1]: lefteye_rect[1] + lefteye_rect[3], lefteye_rect[0]: lefteye_rect[0] + lefteye_rect[2]]
    righteye_rect = cv2.boundingRect(righteye_points) # returns (x,y,w,h) of the rect
    righteye_cropped = res[righteye_rect[1]: righteye_rect[1] + righteye_rect[3], righteye_rect[0]: righteye_rect[0] + righteye_rect[2]]
    #calculate rect area of eye
    # lefteye_rect_area=lefteye_rect[2]*lefteye_rect[3]
    # lefteye_rect_area=cv2.contourArea(np.array([[lefteye_rect[0],lefteye_rect[1]],
    # [lefteye_rect[0]+lefteye_rect[2],lefteye_rect[1]],
    # [lefteye_rect[0]+lefteye_rect[2],lefteye_rect[1]+lefteye_rect[3]],
    # [lefteye_rect[0],lefteye_rect[1]+lefteye_rect[3]]           
    # ]))

    # righteye_rect_area=righteye_rect[2]*righteye_rect[3]
    # calculate Area of each eye 
    lefteye_area=cv2.contourArea( lefteye_points)
    righteye_area=cv2.contourArea( righteye_points)
    if lefteye_area==0 or righteye_area==0:
        pass
    else:
            
        # covered area 
        lefteye_white_area=cv2.countNonZero(lefteye_cropped)          
        righteye_white_area=cv2.countNonZero(righteye_cropped)
        if Eye_cover=="white":
            lefteye_cover_rate=int(lefteye_white_area*100/lefteye_area/1.3)
            righteye_cover_rate=int(righteye_white_area*100/righteye_area/1.3)
        else:
            lefteye_cover_rate=int(100-lefteye_white_area*100/lefteye_area)
            righteye_cover_rate=int(100-righteye_white_area*100/righteye_area)
        
        if lefteye_cover_rate>95:
            lefteye_cover_rate=100
        elif lefteye_cover_rate<25:
            lefteye_cover_rate = 0
        if righteye_cover_rate>95:
            righteye_cover_rate=100
        elif righteye_cover_rate<25:
            righteye_cover_rate=0
        

        cv2.putText(image,"Left eye cover: "+str(lefteye_cover_rate) +"%",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,255),2,cv2.LINE_AA)
        cv2.putText(image,"Right eye cover: "+str(righteye_cover_rate)+"%",(50,130),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,255),2,cv2.LINE_AA)
        # cv2.imshow('Original',res)
        # cv2.imshow("Mask",mask)
        # cv2.imshow("Left eye Cropped", lefteye_cropped )
        # cv2.imshow("Right eye Cropped", righteye_cropped )
        # cv2.imshow("Eye detection", res)
        return lefteye_cover_rate, righteye_cover_rate
def image_eyecover(img_base64, color_th):
    # image = cv2.imread(img_path)
    image = readb64(img_base64)
    image = cv2.flip(image, 1)
    detector = FaceMeshDetector(maxFaces=1)

    image, faces = detector.findFaceMesh(image, True)
    if len(faces)!=0:
        color_th = 80
        lefteye_cover_rate, righteye_cover_rate = eye_coverage_estimate(image, faces, color_th)        
        return  lefteye_cover_rate, righteye_cover_rate

@app.route('/api/predict', methods=['POST'])
def predict():
    if request.method == 'POST':       
        img_base64 = request.json['image_base64']
        color_th = int(request.json['color_th'])        
        lefteye_cover_rate, righteye_cover_rate = image_eyecover(img_base64, color_th)
        # print(lefteye_cover_rate, righteye_cover_rate)
        return jsonify({'lefteye_cover_rate': lefteye_cover_rate, 'righteye_cover_rate': righteye_cover_rate})
if __name__ == "__main__":
    # app.run()
    app.run(host='0.0.0.0', port=8000, debug=True)