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
}

void FaceDetectionFrame::storeLandmarks(std::vector<ofVec2f> new_landmarks)
{
	face_landmarks = new_landmarks;
	has_detected_face = true;
}

void FaceDetectionFrame::setup(int fbo_width, int fbo_height)
{
	camera_texture_fbo.allocate(fbo_width, fbo_height, GL_RGBA, 1);
}
