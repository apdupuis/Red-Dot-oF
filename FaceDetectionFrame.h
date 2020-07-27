#pragma once
#include "ofMain.h"

class FaceDetectionFrame
{
public:
	FaceDetectionFrame();
	~FaceDetectionFrame();

	bool hasFace();
	std::vector<ofVec2f> getFaceLandmarks();
	ofFbo getFbo();
	ofTexture getTexture();
	void storeTexture(ofTexture new_tex);
	void storeLandmarks(std::vector<ofVec2f> new_landmarks);
	void setup(int fbo_width, int fbo_height);
	void printLandmarks();

private:
	ofFbo camera_texture_fbo;
	std::vector<ofVec2f> face_landmarks;
	bool has_detected_face;
};

