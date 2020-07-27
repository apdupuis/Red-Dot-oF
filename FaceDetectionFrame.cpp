#include "FaceDetectionFrame.h"



FaceDetectionFrame::FaceDetectionFrame()
{
	has_detected_face = false;
}


FaceDetectionFrame::~FaceDetectionFrame()
{
}

bool FaceDetectionFrame::hasFace()
{
	return has_detected_face;
}

std::vector<ofVec2f> FaceDetectionFrame::getFaceLandmarks()
{
	return face_landmarks;
}

ofFbo FaceDetectionFrame::getFbo()
{
	return camera_texture_fbo;
}

ofTexture FaceDetectionFrame::getTexture()
{
	return camera_texture_fbo.getTexture();
}

void FaceDetectionFrame::storeTexture(ofTexture new_tex)
{
	camera_texture_fbo.begin();
	ofClear(0);
	new_tex.draw(0, 0);
	camera_texture_fbo.end();

	has_detected_face = false; // when we store new texture we assume no landmarks until they're added
	face_landmarks.clear();
}

void FaceDetectionFrame::storeLandmarks(std::vector<ofVec2f> new_landmarks)
{
	//face_landmarks.clear();
	//for (int i = 0; i < new_landmarks.size(); i++) {
	//	ofVec2f landmark = new_landmarks[i];
	//	face_landmarks.push_back(ofVec2f(landmark.x, landmark.y));
	//}
	face_landmarks = new_landmarks;
	has_detected_face = true;
}

void FaceDetectionFrame::setup(int fbo_width, int fbo_height)
{
	camera_texture_fbo.allocate(fbo_width, fbo_height, GL_RGBA, 1);
}

void FaceDetectionFrame::printLandmarks()
{
	if (has_detected_face) {
		int landmark_len = face_landmarks.size();
		int siz = min(landmark_len, 5);
		for (int i = 0; i < siz; i++) {
			ofVec2f landmark = face_landmarks[i];
			cout << "landmark " << i << ": " << landmark.x << ", " << landmark.y << endl;
		}
	} 
	else {
		cout << "no face" << endl;
	}
}
